# linalgTools.py
#
# Bryan Daniels
# 9.7.2012
# 2024/11/18 forked from neural.linalgTools and neural.LDA
#
# 
#

import numpy as np
import scipy.linalg

# 7.10.2012
def svdInverse(mat,maxEig=1e10,minEig=1e-10,verbose=True):
    u,w,vt = scipy.linalg.svd(mat)
    if any(w==0.):
        raise(ZeroDivisionError, "Singular matrix.")
    wInv = w ** -1
    largeIndices = np.nonzero( abs(wInv) > maxEig )
    wInv[largeIndices] = maxEig*np.sign(wInv[largeIndices])
    
    smallIndices = np.nonzero( abs(wInv) < minEig )
    wInv[smallIndices] = minEig*np.sign(wInv[smallIndices])
    
    if verbose:
        if len(largeIndices) > 0: print("svdInverse:",len(largeIndices),"large singular values out of",len(w))
        if len(smallIndices) > 0: print("svdInverse:",len(smallIndices),"small singular values out of",len(w))
    
    return np.dot( np.dot(vt.T,np.diag(wInv)), u.T )

def LDA(data1,data2,**kwargs):
    """
    Binary Linear Discriminant Analysis 
    (Fisher's Linear Discriminant)
    
    See http://en.wikipedia.org/wiki/Linear_discriminant_analysis
    
    Model two m-dimensional datasets (n1 x m) and (n2 x m) 
    as Gaussians, and return the m-dimensional vector w
    that gives the projection that best separates the two.
    
    3.15.2013 normalized
    """
    mu1 = np.mean(data1,axis=0)
    mu2 = np.mean(data2,axis=0)
    cov1 = np.cov(np.transpose(data1))
    cov2 = np.cov(np.transpose(data2))
    invCov = svdInverse(cov1+cov2,**kwargs)
    
    w = np.dot(invCov,mu2-mu1)
    return w / np.sqrt(np.sum(w**2))
