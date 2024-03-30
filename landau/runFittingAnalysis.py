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
from landauAnalysis import landauAnalysis,gaussianMixtureAnalysis,principalComponents
import time
import sys
import numpy as np
import pandas as pd
import subprocess # for getGitHash
from toolbox import load,save

def loadBeeData(log=True,skipDays=[4,],
    filepath="~/ASUDropbox/Research/bees/geneExpression/Data/170614/nanostring data with VG protein data.xlsx"):
    """
    Returns dictionary keyed by age in days
    """
    xlFile = pd.ExcelFile(filepath)
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

def loadSimulationData(datafile,numSamples,samplesOffset):
    """
    Loads existing simulation data created using runLandauTestSimulations.py
    
    Returns dictionary keyed by mu
    """
    # load data from file created using runLandauTestSimulations.py
    samplesDataDict = load(datafile)
    
    for mu,data in samplesDataDict.items():
        # remove any existing fitting analysis to avoid confusion
        analysisKeys = ['runLandauAnalysis','landauTimeMinutes',
                        'sampleMean','valList','vecList','llList','cList','dList',
                        'simTimeMinutes']
        for k in analysisKeys:
            data.pop(k,None)
        # save simulation gitHash with new name to avoid confusion
        simulationGitHash = data.pop('gitHash',None)
        data['simulationGitHash'] = simulationGitHash
        
        # take desired slice of data
        assert(len(data['finalStates']) >= samplesOffset+numSamples) # enough data?
        data['finalStates'] = data['finalStates'][samplesOffset:samplesOffset+numSamples]
        data['Nsamples'] = numSamples
        
    return samplesDataDict

def runFitting(dataType='bee',ndimsGaussianList=[None,1],
    datafile=None,numSamples=None,samplesOffset=None,outputfilename=None,
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
        dataDictDict = loadSimulationData(datafile,numSamples,samplesOffset)
    else:
        raise(Exception,'Unrecognized dataType')
        
    for key,dataDict in dataDictDict.items():
        
        finalStates = dataDict['finalStates']
        
        if runLandauAnalysis:
            finalStatesLandau = finalStates
            dataOffset = 0.
        
            # run Landau analysis on final states
            startTime = time.time()
            landauOutput = landauAnalysis(finalStatesLandau + dataOffset)
            sampleMean = landauOutput.pop('mu') - dataOffset
            landauTimeMinutes = (time.time() - startTime)/60.
        
            landauOutput.update(
                       {'landauTimeMinutes': landauTimeMinutes,
                        'sampleMean': sampleMean,
                        'gitHash': getGitHash(),
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
                             'gitHash': getGitHash(),
                            } )
                dataDict['gaussianMixtureAnalysis{}'.format(ndims)] = gaussianOutput
            
        if verbose:
            print("runFitting: Done with fitting {} = {}".format(dataVariable,key))
            
        if outputfilename:
            # save data
            save(dataDictDict,outputfilename)
            if verbose:
                print("runFitting: Saved fitting data to {}".format(outputfilename))
                       
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
    dataType = 'simulation' # bee'
    #numNuMax = 1
    ndimsGaussianList = [None,1]
    # parameters for use in sampling from simulation data
    networkName = 'allToAll'
    Ncomponents = 10 #1000 #10 #91
    Nsamples = 16 #100 # number of samples to fit to
    NsamplesOriginal = 1000 #16 #1000 # number of samples in original simulation datafiles
    samplesOffset = 0 #500 # index of first sample used for fitting
    Nmus = 51 #31 #51
    
    # if command line argument is given, use it to set runIndex
    if len(sys.argv) == 2:
        runIndex = int(sys.argv[1])
    elif len(sys.argv) > 2:
        print("Usage: python runFittingAnalysis.py [runIndex]")
        exit()
    else:
        runIndex = 0
    
    # set up output data dictionary and filenames
    if dataType == 'bee':
        filename = 'FittingData_{}.dat'.format(dataType)
        samplesKwargs = {} # (only used for simulation data samples)
    elif dataType == 'simulation':
        filename = 'FittingData_{}_Ncomponents{}_Nsamples{}_offset{}_Nmus{}_run{}.dat'.format(
                networkName,Ncomponents,Nsamples,samplesOffset,Nmus,runIndex)
        datafile = 'LandauTestData_{}_Ncomponents{}_Nsamples{}_Nmus{}_run{}.dat'.format(
                networkName,Ncomponents,NsamplesOriginal,Nmus,runIndex)
        samplesKwargs = {'datafile': datafile,
                         'numSamples': Nsamples,
                         'samplesOffset': samplesOffset,
                        }
    else:
        raise(Exception,"Unrecognized dataType")

    # run the analysis
    dataDictDict = runFitting(dataType,
                          ndimsGaussianList=ndimsGaussianList,
                          runLandauAnalysis=runLandauAnalysis,
                          runGaussianMixtureAnalysis=runGaussianMixtureAnalysis,
                          outputfilename=filename,
                          **samplesKwargs)
    
    # save data
    save(dataDictDict,filename)
    print("runFittingAnalysis: Saved data to {}".format(filename))




        
        
