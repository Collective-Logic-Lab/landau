# landauAnalysis.py
#
# Bryan Daniels
# 2020/11/17
#
# Use analogy with Landau theory of phase transitions to look for
# critical transitions in data.
#

from subprocess import call
import numpy as np
import os

def landauAnalysis(data,numNuMax=10,codeDir='./'):
    """
    Uses Mathematica code to run Landau transition analysis.
    
    Input: Data matrix, shape (#samples)x(#dimensions)
    
    Returns: mu,valList,vecList,llList,cList,dList,nuMuList
    """
    data = np.array(data)
    if len(np.shape(data)) != 2:
        raise TypeError
    dim = len(data[0])
    tempName = "temp_{}".format(os.getpid())
        
    # save data to csv file
    datafile = "{}.csv".format(tempName)
    np.savetxt(datafile,data,delimiter=',')
    
    # call mathematica code
    call([codeDir+"/runLandauTransitionAnalysis.wls",datafile,str(numNuMax)])
    os.remove(datafile)
    
    # read result
    outfile = "{}_LTAoutput.csv".format(tempName)
    try:
        resultList = [ np.loadtxt(outfile,delimiter=',',skiprows=i,max_rows=1) for i in range(8) ]
    except(ValueError):
        print("landauAnalysis: ERROR in Mathematica output.  Returning nans.")
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    mu, valList = resultList[0], resultList[1]
    vecList = [ np.real_if_close(re + (0+1j)*im)
                for re,im in zip(resultList[2].reshape(dim,dim),resultList[3].reshape(dim,dim)) ]
    llList = resultList[4]
    cList, dList, nuMuList = resultList[5], resultList[6], resultList[7]
    os.remove(outfile)
    
    # reshape list arrays to get consistent results when they have length 1
    valList = valList.reshape(valList.size)
    llList = llList.reshape(llList.size)
    cList = cList.reshape(cList.size)
    dList = dList.reshape(dList.size)
    nuMuList = nuMuList.reshape(nuMuList.size)
                            
    return mu,valList,vecList,llList,cList,dList,nuMuList

def principalComponents(data,max_ll_cov=True,reg_covar=1e-6):
    """
    Returns eigenvalues and eigenvectors given by principal components analysis.

    Eigenvectors are returned transposed compared to the numpy convention
    (so that the first eigenvector here is given by vecs[0]).
    
    Note that the eigenvalues returned here correspond to the inverse of
    eigenvalues returned in the landau analysis.
    
    data                    : data should have shape (# samples)x(# dimensions)
    max_ll_cov (True)       : If true, use the maximum likelihood estimate of the
                              covariance matrix (dividing by n).  If false, use the
                              unbiased estimator of the covariance matrix (dividing
                              by n-1).
    reg_covar (1e-6)        : Regularization added to the diagonal of the covariance
                              matrix, as used in sklearn.mixture.GaussianMixture
    """
    # compute covariance matrix
    c = np.cov(data.T,bias=max_ll_cov)
    c += reg_covar*np.eye(len(c))
    
    vals,vecs = np.linalg.eig(c)
    
    return np.real_if_close(vals), [ np.real_if_close(v) for v in vecs.T ]
    
    
