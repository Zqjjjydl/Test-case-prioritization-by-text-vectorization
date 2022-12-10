def getAPFD(TF,n,m):
    ans=1
    for i in range(m):
        ans=ans-1/(n*m)*TF[i]
    ans+=1/(2*n)
    return ans