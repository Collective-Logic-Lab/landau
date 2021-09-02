# analyzeLandauTestSimulations.py
#
# Bryan Daniels
# 2021/7/30
#
# Analyze data from runLandauTestSimulations.py
#

import pandas as pd
import numpy as np
from toolbox.simplePickle import load
import glob

def landauSimulationData(datafilePrefix):

    dataDict = {}
    dfDict = {}
    
    # loop over all files with the given prefix
    fileList = glob.glob("{}*".format(datafilePrefix))
    if len(fileList) == 0:
        raise Exception("No files found with prefix {}".format(datafilePrefix))
    for file in fileList:
        dataDictSingle,dfSingle = landauSimulationData_singleRun(file)
        runIndex = dataDictSingle[list(dataDictSingle.keys())[0]]["runIndex"]
        
        dataDict[runIndex] = dataDictSingle
        dfDict[runIndex] = dfSingle
        
    df = pd.concat(dfDict,names=["runIndex"])
    df = df.sort_values(['mu','runIndex']).reset_index().drop(columns='level_1')
    
    return dataDict,df

def landauSimulationData_singleRun(datafile):
    
    dataDict = load(datafile)
    
    muList = dataDict.keys()
    
    minIndexList = []
    minValList = []
    bistableIndexList = []
    bistableLikelihoodList = []
    bistableCList = []
    bistableDList = []
    bistableEigvalList = []
    for mu in muList:
        if type(dataDict[mu]['llList']) != float:
            # filter out any zero eigenvalues
            nonzeroEigs = np.where(dataDict[mu]['valList'] != 0.)[0]
            dataDict[mu]['valList'] = dataDict[mu]['valList'][nonzeroEigs]
            dataDict[mu]['vecList'] = np.array(dataDict[mu]['vecList'])[nonzeroEigs]
           
            # find dimension with minimum eigenvalue (max variance)
            minIndex = np.argmin(dataDict[mu]['valList'])
            minIndexList.append(minIndex)
            minValList.append(dataDict[mu]['valList'][minIndex])
           
            # find dimension with most evidence for bistability
            bistableIndex = np.argmin(dataDict[mu]['llList'])
            
            # extract bistability parameters for max bistability dimension
            bistableIndexList.append(bistableIndex)
            bistableLikelihoodList.append(dataDict[mu]['llList'][bistableIndex])
            bistableCList.append(dataDict[mu]['cList'][bistableIndex])
            bistableDList.append(dataDict[mu]['dList'][bistableIndex])
            bistableEigvalList.append(dataDict[mu]['valList'][bistableIndex])
        else: # there was an error in the Mathematica code
            bistableIndex = np.nan
            
            bistableIndexList.append(bistableIndex)
            bistableLikelihoodList.append(np.nan)
            bistableCList.append(np.nan)
            bistableDList.append(np.nan)
            bistableEigvalList.append(np.nan)
        
    dfData = {'mu': muList,
              'bistable index': bistableIndexList,
              'bistable log-likelihood': bistableLikelihoodList,
              'bistable c': bistableCList,
              'bistable d': bistableDList,
              'bistable eigenvalue': bistableEigvalList,
              'min eigenvalue index': minIndexList,
              'min eigenvalue': minValList,
              'network name': [dataDict[mu]['networkName'] for mu in muList],
              'Ncomponents': [dataDict[mu]['Ncomponents'] for mu in muList],
              'Nsamples': [dataDict[mu]['Nsamples'] for mu in muList],
              'tFinal': [dataDict[mu]['tFinal'] for mu in muList],
              'simulation time (m)': [dataDict[mu]['simTimeMinutes'] for mu in muList],
              'landau time (m)':  [dataDict[mu]['landauTimeMinutes'] for mu in muList],
              }
    
    df = pd.DataFrame.from_dict(dfData)
    
    return dataDict,df

def runtime(df):
    dftime = df[['runIndex',
                 'simulation time (m)',
                 'landau time (m)']].set_index('runIndex').sum(level='runIndex')
    dftime['total (m)'] = dftime['simulation time (m)'] + dftime['landau time (m)']
    return dftime
