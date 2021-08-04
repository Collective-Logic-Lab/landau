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

def landauSimulationData(datafile):
    
    dataDict = load(datafile)
    
    muList = dataDict.keys()
    
    bistableIndexList = []
    bistableLikelihoodList = []
    bistableCList = []
    bistableDList = []
    bistableEigvalList = []
    for mu in muList:
        if type(dataDict[mu]['llList']) != float:
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
              'network name': [dataDict[mu]['networkName'] for mu in muList],
              'Ncomponents': [dataDict[mu]['Ncomponents'] for mu in muList],
              'Nsamples': [dataDict[mu]['Nsamples'] for mu in muList],
              'tFinal': [dataDict[mu]['tFinal'] for mu in muList],
              'simulation time (m)': [dataDict[mu]['simTimeMinutes'] for mu in muList],
              'landau time (m)':  [dataDict[mu]['landauTimeMinutes'] for mu in muList],
              }
    
    df = pd.DataFrame.from_dict(dfData)
    
    return dataDict,df
