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
from scipy.sparse.linalg import eigs
import os
from sklearn.mixture import GaussianMixture
from scipy.special import gamma,factorial,gammaln
import scipy.optimize

def landauAnalysis(data):
    """
    Run Landau transition analysis.
    
    Input: Data matrix, shape (#samples)x(#dimensions)
    
    Returns: dictionary with mu,valList,vecList,llList,cList,dList,nuMuList,bicDiffList
    
    (Should produce equivalent output to landauAnalysis_mathematica
    with numNuMax=1)
    """
    pass

def landauAnalysis_mathematica(data,numNuMax=1):
    """
    Uses Mathematica code to run Landau transition analysis.
    
    Input: Data matrix, shape (#samples)x(#dimensions)
    
    Returns: dictionary with mu,valList,vecList,llList,cList,dList,nuMuList,bicDiffList
    """
    codeDir = os.path.dirname(os.path.realpath(__file__))
    
    data = np.array(data)
    if len(np.shape(data)) != 2:
        raise TypeError
    numSamples,numDimensions = np.shape(data)
    tempName = "temp_{}".format(os.getpid())
        
    if (numNuMax == 1) and (numDimensions > 1):
        # do dimensionality reduction first if we only want to fit to
        # the first principal component (it's equivalent and makes the
        # mathematica code run much faster)
        vals,vecs = principalComponents(data,k=1)
        sampleMean = np.mean(data,axis=0)
        transformedData = np.dot(data-sampleMean,
                                 np.transpose(vecs))[:,:numNuMax]
        transformedData = np.real_if_close(transformedData)
        # For some reason mathematica chokes when the mean is zero
        # (by definition here).  Add 1 to everything to avoid this
        # (doesn't affect anything except the sample mean mu, which
        # we fix below).
        dataOffset = 1.
        dataForMathematica = transformedData + dataOffset
    else:
        dataOffset = 0.
        dataForMathematica = data + dataOffset
    dim = len(dataForMathematica[0])
        
    # save data to csv file
    datafile = "{}/{}.csv".format(codeDir,tempName)
    np.savetxt(datafile,dataForMathematica,delimiter=',')
    
    # call mathematica code
    call(["./runLandauTransitionAnalysis.wls",datafile,str(numNuMax)],
        cwd=codeDir)
    os.remove(datafile)
    
    # read result
    outfile = "{}/{}_LTAoutput.csv".format(codeDir,tempName)
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
    mu = resultList[0]
    llList = resultList[4]
    cList, dList, nuMuList = resultList[5], resultList[6], resultList[7]
    valList = resultList[1]
    vecList = [ np.real_if_close(re + (0+1j)*im)
                for re,im in zip(resultList[2].reshape(dim,dim),resultList[3].reshape(dim,dim)) ]
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
                            
    return {'mu': mu - dataOffset,
            'valList': valList,
            'vecList': vecList,
            'llList': llList,
            'cList': cList,
            'dList': dList,
            'nuMuList': nuMuList,
            'bicDiffList': bicDiffList,
            }

def principalComponents(data,max_ll_cov=True,reg_covar=1e-6,k=None,seed=1234):
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
    k (None)                : Number of principal components to compute.  If None,
                              defaults to all principal components (# dimensions)
    seed (1234)             : Used to set the random number generator for
                              initialization of scipy.sparse.linalg.eigs in
                              the case that k<(# dimensions)
    """
    # compute covariance matrix
    c = np.cov(data.T,bias=max_ll_cov)
    c += reg_covar*np.eye(len(c))
    
    if (k is None) or (k == len(c)):
        vals,vecs = np.linalg.eig(c)
    elif k < len(c): # use scipy.sparse.linalg.eigs for speed when k<(# dimensions)
        np.random.seed(seed)
        v0 = np.random.rand(len(c))
        vals,vecs = eigs(c,k=k,v0=v0)
    else:
        raise(Exception,"k cannot be greater than the width of the data")
    
    return np.real_if_close(vals), [ np.real_if_close(v) for v in vecs.T ]
    
def LandauTransitionDistributionRelativeLogPDF_multiple_dimensions(x, mu, Jvals, Jvecs, nu, c, d):
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

def LandauTransitionDistributionRelativeLogPDF(x, mu, Jnu, c, h, d):
    """
    x should be a single number
    
    (Note: mu here is the mean, not the interaction strength)
    """
    assert(len(np.shape(x))==0)
    
    term1 = -(1./2.)* Jnu * (x - mu)**2
    term2 = - ((c - 1.)/2.) * Jnu * (x - mu)**2
    term3 = - (h/3.) * Jnu**(3/2) * (x - mu)**3
    term4 = - (d/4.) * Jnu**2 * (x - mu)**4
    
    return term1 + term2 + term3 + term4

def logNormalizationGaussian(Jnu):
    """
    Normalization factor to turn LandauTransitionDistributionRelativeLogPDF
    with c = 1, h = d = 0, into a normalized Gaussian distribution
    
    Jnu = 1 / variance
    """
    return 0.5 * np.log(2*np.pi) - 0.5 * np.log(Jnu)

def GaussianDistributionLogPDF(x, mu, Jnu):
    rellogpdf = LandauTransitionDistributionRelativeLogPDF(x,mu,Jnu,1,0,0)
    Z = logNormalizationGaussian(Jnu)
    return rellogpdf - Z

def normalizationZ_old(Jnu,c,h,d,maxorder=30):
    """
    Series estimate of the normalization factor

    \int_{-\infty}^{\infty} \exp( LandauTransitionDistributionRelativeLogPDF ) dx
    
    Does not use logs and is therefore limited to relatively small orders
    """
    # define ranges of n and m to sum over
    n_list = np.arange(maxorder//2)
    m_list = np.arange(maxorder)

    # define all combinations of n and m using 2D grids
    n,m = np.meshgrid(n_list,m_list)

    # compute factors for series
    factor1 = ((-h*Jnu**(3/2)/3)**(2*n)) / factorial(2*n)
    factor2 = ((-c*Jnu/2)**m) / factorial(m)
    factor3 = gamma((6*n + 2*m + 1)/4) / ( ((d*Jnu**2)/4)**((6*n + 2*m + 1)/4) )
    summand_mat = factor1 * factor2 * factor3
    result = 0.5 * np.sum(summand_mat)

    # check for convergence: large n and m should be adding small corrections
    result_smaller_order = 0.5*(np.sum(summand_mat[:-1,:-1]))
    if abs(result_smaller_order-result)/abs(result) > 1e-5:
        print("normalizationZ: WARNING: lack of convergence for Jnu = {}, c = {}, h = {}, d = {}, maxorder = {}".format(
            Jnu,c,h,d,maxorder))
    
    return result
    
def normalizationZ(Jnu,c,h,d,maxorder=300):
    """
    Series estimate of the normalization factor

    \int_{-\infty}^{\infty} \exp( LandauTransitionDistributionRelativeLogPDF ) dx
    """
    assert(d>=0)
    
    # define ranges of n and m to sum over
    n_list = np.arange(maxorder//2)
    m_list = np.arange(maxorder)

    # define all combinations of n and m using 2D grids
    n,m = np.meshgrid(n_list,m_list)

    # compute factors for series
    sign_factor1 = np.sign((-h)**((2*n)%2))
    sign_factor2 = np.sign((-c)**(m%2))
    log_factor1 = 2*n * np.log(np.abs(h)*Jnu**(3/2)/3) - gammaln(2*n+1)
    log_factor2 = m * np.log(np.abs(c)*Jnu/2) -  gammaln(m+1)
    log_factor3 = gammaln((6*n + 2*m + 1)/4) \
                  - ((6*n + 2*m + 1)/4)*np.log((d*Jnu**2)/4)
    
    summand_sign = sign_factor1 * sign_factor2
    summand_mat = summand_sign*np.exp(log_factor1 + log_factor2 + log_factor3)
    result = 0.5 * np.sum(summand_mat)

    # check for convergence: large n and m should be adding small corrections
    result_smaller_order = 0.5*(np.sum(summand_mat[:-1,:-1]))
    if abs(result_smaller_order-result)/abs(result) > 1e-5:
        print("normalizationZ: WARNING: lack of convergence for Jnu = {}, c = {}, h = {}, d = {}, maxorder = {}".format(
            Jnu,c,h,d,maxorder))
    
    return result
    
def log_likelihood(x_list,c,h,d,numu,maxorder=300):
    """
    x_list should be a list of single numbers of length num_samples
    """
    mu = np.mean(x_list)
    Jnu = 1./np.var(x_list)
    Z = normalizationZ(Jnu,c,h,d,maxorder)
    log_likelihoods = [ LandauTransitionDistributionRelativeLogPDF(x,mu+numu,Jnu,c,h,d) - np.log(Z) for x in x_list ]
    return np.sum(log_likelihoods)
    
def log_likelihood_difference_from_gaussian(x_list,c,h,d,numu,maxorder=300):
    """
    x_list should be a list of single numbers of length num_samples
    """
    mu = np.mean(x_list)
    Jnu = 1./np.var(x_list)
    landau_ll = log_likelihood(x_list,c,h,d,numu,maxorder)
    gaussian_ll = np.sum([ GaussianDistributionLogPDF(x,mu,Jnu) for x in x_list ])
    return landau_ll - gaussian_ll
    
def maximize_landau_log_likelihood(x_list,cinit=-1,hinit=1e-3,
    dinit=2,numuinit=0,maxorder=500,cmax=1.5,abs_hmax=None,dmin=0.1):
    """
    Find maximum likelihood fit to the landau distribution including bias.
    """
    func = lambda params: -log_likelihood_difference_from_gaussian(x_list,
        *params,maxorder=maxorder)
    
    # set parameter bounds
    if abs_hmax is None:
        hbounds = (None,None)
    else:
        hbounds = (-abs_hmax,abs_hmax)
    cbounds,dbounds,numubounds = (None,cmax),(dmin,None),(None,None)
    
    # do optimization
    return scipy.optimize.minimize(func,(cinit,hinit,dinit,numuinit),
        bounds=(cbounds,hbounds,dbounds,numubounds))
    
def gaussianMixtureAnalysis(data,ndims=None,cov_type='tied',nclusters=2,
    returnFittingObjects=False,**kwargs):
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
    returnFittingObjects (False): If True, also return the sklearn objects
                                  representing the fit distributions.
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
    
    if returnFittingObjects:
        return {'llDiff': llDiff,
                'bicDiff': bicDiff,
                'gSingle': gSingle,
                'gMultiple': gMultiple,
                }
    else:
        return {'llDiff': llDiff,
                'bicDiff': bicDiff,
                }
    
