import sys

import numpy as np
import random
import os
from model import getOrderSentenceBert,getOrderLDA,getOrderCodeBert,getOrderProdLDA


def readRaw(d):
    faultMatrix=[]
    f=open("../input/"+d+"/test_matrix.dat")
    lines=f.readlines()
    for l in lines:
        vec=l.strip().split(' ')
        vec=[int(v) for v in vec]
        faultMatrix.append(vec)
    testCaseNames=os.listdir("../input/"+d+"/raw")
    testCases=[]
    for n in testCaseNames:
        testCase_f=open("../input/"+d+"/raw/"+n)
        testCase=testCase_f.read()
        import re
        testCase = re.sub(r'/\*(.|[\r\n])*?\*/', "", testCase)
        testCases.append(testCase)
    return faultMatrix,testCases

dataset=["antv7","derbyv1","derbyv2","derbyv3","derbyv5"]


def readBow(d):
    faultMatrix=[]
    f=open("../input/"+d+"/test_matrix.dat")
    lines=f.readlines()
    for l in lines:
        vec=l.strip().split(' ')
        vec=[int(v) for v in vec]
        faultMatrix.append(vec)
    testCaseNames=os.listdir("../input/"+d+"/preA.pruned0180")
    testCases=[]
    for n in testCaseNames:
        testCase_f=open("../input/"+d+"/preA.pruned0180/"+n)
        testCaseLines=testCase_f.readlines()
        testCase=""
        for l in testCaseLines:
            testCase=testCase+l.strip()+" "
        testCase=testCase.strip()
        testCases.append(testCase)
    return faultMatrix,testCases

def getAPFD(TF,n,m):
    ans=1
    for i in range(m):
        ans=ans-(1/(n*m))*TF[i]
    ans+=1/(2*n)
    return ans

if __name__ == "__main__":
    algo = sys.argv[1]
    for d in dataset:
        faultMatrix, testCases = readRaw(d)
        order = []
        if algo == "codeBERT":
            order, embeddings = getOrderCodeBert(testCases)
        elif algo == "LDA":
            order = getOrderLDA(testCases)
        else:
            print("Wrong Parameter")
            exit()

        faultMatrix = [faultMatrix[i] for i in order]

        n = len(faultMatrix)
        m = len(faultMatrix[0])
        TF = []

        for j in range(m):  # fault
            for i in range(n):  # case
                if faultMatrix[i][j] == 1:
                    TF.append(i + 1)
                    break
        APFD = getAPFD(TF, n, m)
        print("APFD for " + d + " :", APFD)

