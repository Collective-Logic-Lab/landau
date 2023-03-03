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
from sklearn.mixture import GaussianMixture

def landauAnalysis(data,numNuMax=10,codeDir='./'):
    """
    Uses Mathematica code to run Landau transition analysis.
    
    Input: Data matrix, shape (#samples)x(#dimensions)
    
    Returns: dictionary with mu,valList,vecList,llList,cList,dList,nuMuList,bicDiffList
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
        return {'mu': np.nan,
                'valList': np.nan,
                'vecList': np.nan,
                'llList': np.nan,
                'cList': np.nan,
                'dList': np.nan,
                'nuMuList': np.nan,
                'bicDiffList': np.nan,
                }
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
                            
    # also return bic differences
    numExtraParameters = 1
    bicDiffList = 2.*llList + numExtraParameters*np.log(len(data))
                            
    return {'mu': mu,
            'valList': valList,
            'vecList': vecList,
            'llList': llList,
            'cList': cList,
            'dList': dList,
            'nuMuList': nuMuList,
            'bicDiffList': bicDiffList,
            }

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
    
def LandauTransitionDistributionRelativeLogPDF(x, mu, Jvals, Jvecs, nu, c, d):
    """
    x should have length (#dimensions)
    
    (Note: mu here is the mean, not the interaction strength)
    """
    assert(len(nu) == len(Jvecs[0]))
    assert(len(Jvals) == len(Jvecs))
    assert(len(np.shape(x))==1)
    assert(len(x) == len(nu))
    
    Jvecs = np.array(Jvecs)
    
    # J has shape (#dimensions)x(#dimensions)
    #J = Transpose[Jvecs].DiagonalMatrix[Jvals].Jvecs,
    J = np.dot(np.dot(Jvecs.T,np.diag(Jvals)),Jvecs)
    
    nuJnu = np.dot(np.dot(nu,J),nu)
    term1 = -(1./2.)* np.dot(np.dot(x - mu,J),x - mu)
    term2 = - ((c - 1.)/2.) * nuJnu * np.dot(x - mu,nu)**2
    term3 = - (d/4.) * nuJnu**2 * np.dot(x - mu,nu)**4
    
    return term1 + term2 + term3
    
def gaussianMixtureAnalysis(data,ndims=None,cov_type='tied',nclusters=2,**kwargs):
    """
    Compares a gaussian mixture model with nclusters clusters (default 2) to
    a simple single gaussian.
    
    Returns difference in log-likelihoods and sklearn fit GaussianMixture object
    corresponding to the multiple clusters model.
    
    See sklearn.mixture.GaussianMixture for other kwargs, including n_init
    (these kwargs are passed to both the single and multiple cluster models).
    
    data                        : data should have shape (# samples)x(# dimensions)
    ndims (None)                : Number of highest-variance dimensions to include
                                  in the analysis.  The default (None) corresponds
                                  to keeping all dimensions.
    cov_type ('tied')           : Can be {'full', 'tied', 'diag', 'spherical'}.
                                  See sklearn.mixture.GaussianMixture.
    """
    
    # optionally reduce dimensionality
    if ndims:
        mu = np.mean(data,axis=0)
        vals,vecs = principalComponents(data)
        transformedData = np.dot(data-mu,np.transpose(vecs))[:,:ndims]
        transformedData = np.real_if_close(transformedData)
    else:
        transformedData = data
    
    # perform fits
    gMultiple = GaussianMixture(n_components=nclusters,
                                covariance_type=cov_type,
                                **kwargs).fit(transformedData)
    gSingle = GaussianMixture(n_components=1,
                              covariance_type=cov_type,
                              **kwargs).fit(transformedData)
               
    # calculate difference in log-likelihoods
    # (this is analogous to what the mathematica landau code produces)
    llMultiple = gMultiple.score(transformedData) * len(transformedData)
    llSingle = gSingle.score(transformedData) * len(transformedData)
    llDiff = llSingle - llMultiple
    
    # use bic instead...
    # (lower bic = better, so negative bicDiff indicates evidence pointing toward
    #  multiple gaussian model)
    bicSingle = gSingle.bic(transformedData)
    bicMultiple = gMultiple.bic(transformedData)
    bicDiff = bicMultiple - bicSingle
    
    return {'llDiff': llDiff,
            'bicDiff': bicDiff,
            'gSingle': gSingle,
            'gMultiple': gMultiple,
            }
    
