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
        # find dimension with most evidence for bistability
        bistableIndex = np.argmin(dataDict[mu]['llList'])
        
        # extract bistability parameters for max bistability dimension
        bistableIndexList.append(bistableIndex)
        bistableLikelihoodList.append(dataDict[mu]['llList'][bistableIndex])
        bistableCList.append(dataDict[mu]['cList'][bistableIndex])
        bistableDList.append(dataDict[mu]['dList'][bistableIndex])
        bistableEigvalList.append(dataDict[mu]['valList'][bistableIndex])
        
    dfData = {'mu': muList,
              'bistable index': bistableIndexList,
              'bistable log-likelihood': bistableLikelihoodList,
              'bistable c': bistableCList,
              'bistable d': bistableDList,
              'bistable eigenvalue': bistableEigvalList,
              'network name': [dataDict[mu]['networkName'] for mu in muList],
              'simulation time (m)': [dataDict[mu]['simTimeMinutes'] for mu in muList],
              'landau time (m)':  [dataDict[mu]['landauTimeMinutes'] for mu in muList],
              }
    
    df = pd.DataFrame.from_dict(dfData)
    
    return dataDict,df
