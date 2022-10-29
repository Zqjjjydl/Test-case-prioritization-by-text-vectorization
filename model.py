from scipy import spatial
from sentence_transformers import SentenceTransformer
from random import randrange
import math
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

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

def getDistanceLDA(sentences):
    sentences=[s.split(" ") for s in sentences]
    common_dictionary = Dictionary(sentences)
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    lda = LdaModel(common_corpus, num_topics=10)
    embeddings = []
    for doc in common_corpus:
        embeddings.append(lda[doc])

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
    order=greedySearch(sentences,distances)
    
    return order

def getOrderLDA(sentences):

    distances=getDistanceLDA(sentences)
    order=greedySearch(sentences,distances)
    
    return order