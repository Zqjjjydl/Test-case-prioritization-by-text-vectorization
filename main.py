import numpy as np
import random
import os
from model import getOrderSentenceBert

dataset=["antv7","derbyv1","derbyv2","derbyv3","derbyv5"]

def getAPFD(TF,n,m):
    ans=1
    for i in range(m):
        ans=ans-(1/(n*m))*TF[i]
    ans+=1/(2*n)
    return ans

for d in dataset:
    faultMatrix=[]
    f=open("./"+d+"/test_matrix.dat")
    lines=f.readlines()
    for l in lines:
        vec=l.strip().split(' ')
        vec=[int(v) for v in vec]
        faultMatrix.append(vec)
    testCaseNames=os.listdir("./"+d+"/preA.pruned0180")
    testCases=[]
    for n in testCaseNames:
        testCase_f=open("./"+d+"/preA.pruned0180/"+n)
        testCaseLines=testCase_f.readlines()
        testCase=""
        for l in testCaseLines:
            testCase=testCase+l.strip()+" "
        testCase=testCase.strip()
        testCases.append(testCase)
    order=getOrderSentenceBert(testCases)
    faultMatrix=[faultMatrix[i] for i in order]

    n=len(faultMatrix)
    m=len(faultMatrix[0])
    TF=[]
    
    for j in range(m):#fault
        for i in range(n):#case
            if faultMatrix[i][j]==1:
                TF.append(i+1)
                break
    APFD=getAPFD(TF,n,m)
    print("APFD for "+d+" :",APFD)
    #text to vector

#calculate matric

