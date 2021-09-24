# runFittingAnalysis.py
#
# Bryan Daniels
# 2021/9/24
#
# Branched from runLandauTestAnalysis.py
#
# Run data fitting (Landau and/or Gaussian mixture model) and save the results.
#

from criticalDynamics import simpleCollectiveDynamics,allToAllNetworkAdjacency
from landauAnalysis import landauAnalysis,gaussianMixtureAnalysis
import time
import sys
import numpy as np
import pandas as pd
import subprocess # for getGitHash
from toolbox.simplePickle import save

def loadBeeData(log=True,skipDays=[4,]):
    """
    Returns dictionary keyed by age in days
    """
    xlFile = pd.ExcelFile("../Data/170614/nanostring data with VG protein data.xlsx")
    rawData = xlFile.parse('Sheet1')
    speciesNames = [name for name in rawData.keys()][4:-1]
    ages = np.unique(rawData['Age'])
    #print("Number of species = {}".format(len(speciesNames)))
    dataFull = rawData.set_index('Age')
    dataFull.drop(['Unnamed: 2','Unnamed: 3','Gene','Sample ID'],axis=1,inplace=True)
    
    if log:
        data = np.log(dataFull)
    else:
        data = dataFull
        
    return dict([ (age,data.loc[age]) for age in ages if age not in skipDays ])
 
def loadBeeDataDict(log=True):
    """
    Returns dictionary of dictionaries of form for use in runFitting
    """
    
    beeData = loadBeeData(log=log)
    
    dataDictDict = {}
    for age,data in beeData.items():
        dataDictDict[age] = {'age':age,
                             'log':log,
                             'finalStates':data,
                            }
    return dataDictDict

def loadSimulationData():
    """
    Returns dictionary keyed by mu
    """

def runFitting(dataType='bee',numNuMax=10,ndimsGaussianList=[None,1],
    runLandauAnalysis=True,runGaussianMixtureAnalysis=True,
    verbose=True):
    """
    dataType ('bee')        : One of ['bee','simulation']
    """
        
    # load data
    if dataType == 'bee':
        dataVariable = 'age'
        dataDictDict = loadBeeDataDict()
    elif dataType == 'simulation':
        dataVariable = 'mu'
        dataDictDict = XXX
    else:
        raise(Exception,'Unrecognized dataType')
        
    for key,dataDict in dataDictDict.items():
        
        finalStates = dataDict['finalStates']
        
        if runLandauAnalysis:
            # run Landau analysis on final states
            startTime = time.time()
            landauOutput = landauAnalysis(finalStates,numNuMax=numNuMax)
            sampleMean = landauOutput.pop('mu')
            landauTimeMinutes = (time.time() - startTime)/60.
        
            landauOutput.update(
                       {'landauTimeMinutes': landauTimeMinutes,
                        'numNuMax': numNuMax,
                        'sampleMean': sampleMean,
                       } )
            dataDict['landauAnalysis'] = landauOutput
                       
        if runGaussianMixtureAnalysis:
            # run Gaussian Mixture analysis on final states
            for ndims in ndimsGaussianList:
                startTime = time.time()
                gaussianOutput = gaussianMixtureAnalysis(finalStates,ndims=ndims)
                gaussianTimeMinutes = (time.time() - startTime)/60.
                
                gaussianOutput.update(
                            {'gaussianTimeMinutes': gaussianTimeMinutes,
                             'ndims': ndims,
                            } )
                dataDict['gaussianMixtureAnalysis{}'.format(ndims)] = gaussianOutput
            
        if verbose:
            print("runFitting: Done with fitting {} {}".format(dataVariable,key))
                       
    return dataDictDict
        
def getGitHash(dir='./'):
    """
    Get the hash code for the current HEAD state of the git repository
    in directory 'dir'.
    """
    hashVal = subprocess.check_output(['git','rev-parse','HEAD'],cwd=dir)
    return hashVal.strip().decode()

if __name__ == '__main__':

    # set up parameters of run
    runLandauAnalysis = True
    runGaussianMixtureAnalysis = True
    dataType = 'bee' # 'simulation'
    numNuMax = 1
    ndimsGaussianList = [None,1]
    
    baseDict = {'dataType': dataType,
                'gitHash': getGitHash(),
                'runIndex': runIndex,
                'runLandauAnalysis': runLandauAnalysis,
                'runGaussianMixtureAnalysis': runGaussianMixtureAnalysis,
               }

    # run the analysis
    dataDictDict = runFitting(dataType,
                          numNuMax=numNuMax,
                          ndimsGaussianList=ndimsGaussianList,
                          runLandauAnalysis=runLandauAnalysis,
                          runGaussianMixtureAnalysis=runGaussianMixtureAnalysis)
    
    # save data
    filename = 'FittingData_{}_run{}.dat'.format(
                dataType,runIndex)
    save(dataDictDict,filename)
    print("runFittingAnalysis: Saved data to {}".format(filename))




        
        
