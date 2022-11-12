from scipy import spatial
from sentence_transformers import SentenceTransformer
from random import randrange
import math
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import argparse
from tqdm import tqdm

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from dataloader import trainDataset
from torch.utils.data import Dataset, DataLoader


#prodLDA是抄的
class ProdLDA(nn.Module):
    def __init__(self, net_arch):
        super(ProdLDA, self).__init__()
        ac = net_arch
        self.net_arch = net_arch
        # encoder
        self.en1_fc     = nn.Linear(ac.num_input, ac.en1_units)
        self.en2_fc     = nn.Linear(ac.en1_units, ac.en2_units)
        self.en2_drop   = nn.Dropout(0.2)
        self.mean_fc    = nn.Linear(ac.en2_units, ac.num_topic)
        self.mean_bn    = nn.BatchNorm1d(ac.num_topic)               # bn for mean
        self.logvar_fc  = nn.Linear(ac.en2_units, ac.num_topic)
        self.logvar_bn  = nn.BatchNorm1d(ac.num_topic)               # bn for logvar
        # z
        self.p_drop     = nn.Dropout(0.2)
        # decoder
        self.topic_emb  = nn.Linear(ac.num_topic, ac.dh, bias=False)
        self.word_emb = nn.Linear(ac.dh, ac.num_input)
        self.decoder_bn = nn.BatchNorm1d(ac.num_input)           # bn for decoder

        # prior mean and variance as constant buffers
        prior_mean   = torch.Tensor(1, ac.num_topic).fill_(0)
        prior_var    = torch.Tensor(1, ac.num_topic).fill_(ac.variance)
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)

        # initial
        nn.init.xavier_normal(self.en1_fc.weight, 1)
        nn.init.xavier_normal(self.en2_fc.weight, 1)
        nn.init.xavier_normal(self.mean_fc.weight, 1)
        nn.init.xavier_normal(self.logvar_fc.weight, 1)
        nn.init.xavier_normal(self.topic_emb.weight, 1)
        nn.init.xavier_normal(self.word_emb.weight, 1)
        nn.init.constant(self.en1_fc.bias, 0.0)
        nn.init.constant(self.en2_fc.bias, 0.0)
        nn.init.constant(self.mean_fc.bias, 0.0)
        nn.init.constant(self.logvar_fc.bias, 0.0)
        nn.init.constant(self.word_emb.bias, 0.0)

    def forward(self, input, compute_loss, avg_loss):
        # compute posterior
        en1 = F.softplus(self.en1_fc(input))
        en2 = F.softplus(self.en2_fc(en1))
        en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn(self.mean_fc  (en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        posterior_var    = posterior_logvar.exp()
        # take sample
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        p = F.softmax(z, dim=1)                                         # mixture probability
        p = self.p_drop(p)
        # use p for doing reconstruction
        dt_vec = F.tanh(self.topic_emb(p))
        recon = F.softmax(self.decoder_bn(self.word_emb(dt_vec)), dim=1)   # reconstructed distribution over vocabulary

        if compute_loss:
            return recon, self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss), dt_vec
        else:
            return recon, dt_vec

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL  = -(input * (recon+1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017, 
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.net_arch.num_topic)
        # loss
        loss = (NL + KLD)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss

def trainProdLDA(model):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--en1-units',        type=int,   default=100)
    parser.add_argument('-s', '--en2-units',        type=int,   default=100)
    parser.add_argument('-dt', '--dt',        type=int,   default=300)
    parser.add_argument('-dw', '--dw',        type=int,   default=300)
    parser.add_argument('-dh', '--dh',        type=int,   default=300)
    parser.add_argument('-t', '--num-topic',        type=int,   default=30)
    parser.add_argument('-em', '--num-emotion',        type=int,   default=6)
    parser.add_argument('-b', '--batch-size',       type=int,   default=20)
    parser.add_argument('-o', '--optimizer',        type=str,   default='Adam')
    parser.add_argument('-r', '--learning-rate',    type=float, default=0.003)
    parser.add_argument('-wd', '--weight-decay',    type=float, default=5e-5)
    parser.add_argument('-lam', '--lambd',    type=float, default=0.03)
    parser.add_argument('-sl', '--max-sentencelen',         type=int, default=19)
    parser.add_argument('-e', '--num-epoch',        type=int,   default=800)
    parser.add_argument('-q', '--init-mult',        type=float, default=1.0)    # multiplier in initialization of decoder weight
    parser.add_argument('-v', '--variance',         type=float, default=0.995)  # default variance in prior normal
    parser.add_argument('--nogpu',                  action='store_true')        # do not use GPU acceleration
    parser.add_argument('-m', '--momentum',         type=float, default=0.99)
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    trainDataDealer = trainDataset(args)
    train_loader = DataLoader(dataset=trainDataDealer,
                          batch_size=args.batch_size,
                          shuffle=False)
    model=ProdLDA(model)
    model=model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay,betas=(args.momentum, 0.999))
    progressive = tqdm(range(args.num_epoch), total=args.num_epoch,
                   ncols=50, leave=False, unit="b")
    for epoch in progressive:
        loss_list=[]
        #train
        model.train()
        sentence_count=0
        model.zero_grad()
        for i, data in enumerate(train_loader):

            bow,dataInIdx,target=data
            bow=bow.to(device)
            bow=bow.double()
            dataInIdx=dataInIdx.to(device)
            target=target.to(device)
            
            l_ntm=model(bow,True)

            recon,vt,loss=l_ntm
            # +args.lambd*l_ntm

            loss_list.append(loss.item())
            loss.to(device)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        avg_loss=torch.mean(torch.tensor(loss_list)).item()
        print("avg_loss",avg_loss)
    return model

def angularDis(v1,v2):
    cosSim = 1 - spatial.distance.cosine(v1,v2)
    dis=math.acos(cosSim)/math.pi
    return dis

def getDistanceSentenceBert(sentences):
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    embeddings = model.encode(sentences)

    distances=[[0 for j in range(len(sentences))] for i in range(len(sentences))]
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            distances[i][j]=angularDis(embeddings[i],embeddings[j])
            distances[j][i]=distances[i][j]
    return distances

def getDistanceProdLDA(sentences):
    sentences=[s.split(" ") for s in sentences]
    common_dictionary = Dictionary(sentences)
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    lda = ProdLDA()
    lda=trainProdLDA(lda,common_corpus)
    embeddings = []
    for doc in common_corpus:
        embeddings.append(lda[doc])

    distances=[[0 for j in range(len(sentences))] for i in range(len(sentences))]
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            distances[i][j]=angularDis(embeddings[i],embeddings[j])
            distances[j][i]=distances[i][j]
    return distances

def getDistanceLDA(sentences):
    sentences=[s.split(" ") for s in sentences]
    common_dictionary = Dictionary(sentences)
    common_corpus = [common_dictionary.doc2bow(text) for text in sentences]
    lda = LdaModel(common_corpus, num_topics=10)
    embeddings = []
    for doc in common_corpus:
        tempEmbed=lda.get_document_topics(doc,minimum_probability=0)
        tempEmbed=[i[-1] for i in tempEmbed]
        embeddings.append(tempEmbed)

    distances=[[0 for j in range(len(sentences))] for i in range(len(sentences))]
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            distances[i][j]=angularDis(embeddings[i],embeddings[j])
            distances[j][i]=distances[i][j]
    return distances

def greedySearch(distances):
    order=[randrange(0,len(distances))]
    remainIdx=[i for i in range(len(distances)) if i not in order]
    while len(remainIdx)>0:
        maxIdx=remainIdx[0]
        maxDis=float('-inf')
        for i in remainIdx:#for all remaining point find lartgest
            minIdx=remainIdx[0]#distance with order
            minDis=float('inf')
            for j in order:#for point in order
                if distances[i][j]<minDis:
                    minIdx=j
                    minDis=distances[i][j]
            if minDis>maxDis:
                maxDis=minDis
                maxIdx=i
        order.append(maxIdx)
        remainIdx.remove(maxIdx)
    return order

def getOrderSentenceBert(sentences):

    distances=getDistanceSentenceBert(sentences)
    order=greedySearch(distances)
    
    return order

def getOrderLDA(sentences):

    distances=getDistanceLDA(sentences)
    order=greedySearch(distances)
    
    return order