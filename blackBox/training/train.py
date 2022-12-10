import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from model import BertForCL

from parameter import ModelArguments, DataTrainingArguments, OurTrainingArguments
logger = logging.getLogger(__name__)

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file#java_train.txt
extension = "text"
datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")

config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
if model_args.config_name:
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
elif model_args.model_name_or_path:
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }   
if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
elif model_args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
model.resize_token_embeddings(len(tokenizer))
column_names = datasets["train"].column_names
sent2_cname = None
if len(column_names) == 2:
    # Pair datasets
    sent0_cname = column_names[0]
    sent1_cname = column_names[1]
elif len(column_names) == 3:
    # Pair datasets with hard negatives
    sent0_cname = column_names[0]
    sent1_cname = column_names[1]
    sent2_cname = column_names[2]
elif len(column_names) == 1:
    # Unsupervised datasets
    sent0_cname = column_names[0]
    sent1_cname = column_names[0]
else:
    raise NotImplementedError

def prepare_features(examples):
    # padding = longest (default)
    #   If no sentence in the batch exceed the max length, then use
    #   the max sentence length in the batch, otherwise use the 
    #   max sentence length in the argument and truncate those that
    #   exceed the max length.
    # padding = max_length (when pad_to_max_length, for pressure test)
    #   All sentences are padded/truncated to data_args.max_seq_length.
    total = len(examples[sent0_cname])

    # Avoid "None" fields 
    for idx in range(total):
        if examples[sent0_cname][idx] is None:
            examples[sent0_cname][idx] = " "
        if examples[sent1_cname][idx] is None:
            examples[sent1_cname][idx] = " "
    
    sentences = examples[sent0_cname] + examples[sent1_cname]

    # If hard negative exists
    if sent2_cname is not None:
        for idx in range(total):
            if examples[sent2_cname][idx] is None:
                examples[sent2_cname][idx] = " "
        sentences += examples[sent2_cname]

    sent_features = tokenizer(
        sentences,
        max_length=data_args.max_seq_length,
        truncation=True,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    features = {}
    if sent2_cname is not None:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
    else:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
        
    return features
if training_args.do_train:
    train_dataset = datasets["train"].map(
        prepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

data_collator = default_data_collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.model_args = model_args

# Training
if training_args.do_train:
    model_path = (
        model_args.model_name_or_path
        if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
        else None
    )
    train_result = trainer.train(model_path=model_path)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
    if trainer.is_world_process_zero():
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

# Evaluation
results = {}
if training_args.do_eval:
    logger.info("*** Evaluate ***")
    results = trainer.evaluate(eval_senteval_transfer=True)

    output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    if trainer.is_world_process_zero():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in sorted(results.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")






# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
# from transformers import AutoModelForSequenceClassification
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
# for name, param in model.named_parameters():
#     print(name)

# from transformers import TrainingArguments
# training_args = TrainingArguments(output_dir="test_trainer")
# import numpy as np
# import evaluate
# metric = evaluate.load("accuracy")
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
# from transformers import TrainingArguments, Trainer

# training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )
# trainer.train()