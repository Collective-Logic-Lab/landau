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
    
    Returns: mu,valList,vecList,llList,cList,dList
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
        resultList = [ np.loadtxt(outfile,delimiter=',',skiprows=i,max_rows=1) for i in range(7) ]
    except(ValueError):
        print("landauAnalysis: ERROR in Mathematica output.  Returning nans.")
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    mu, valList = resultList[0], resultList[1]
    vecList = [ np.real_if_close(re + (0+1j)*im)
                for re,im in zip(resultList[2].reshape(dim,dim),resultList[3].reshape(dim,dim)) ]
    llList, cList, dList = resultList[4], resultList[5], resultList[6]
    os.remove(outfile)
                            
    return mu,valList,vecList,llList,cList,dList
