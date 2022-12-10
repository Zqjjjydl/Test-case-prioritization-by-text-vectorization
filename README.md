# Test-case-prioritization-by-text-vectorization
Project for software testing course

To run CodeBERT text case prioritization, download pretrained model, unzip it and put it under folder `model`. Then run:

```
python blackbox.py
```

The pretrained model has been sent to TA via email because of its large size.

Also, other black box baseline methods can be run by changing line 55 in blackbox.py

To run white box baseline methods, run:

```
python ./whiteBox/prioritize.py
```

To train a new CodeBERT, run:

```
sh ./training/run_unsup_example.sh
```



