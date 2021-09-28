# analyzeLandauTestSimulations.py
#
# Bryan Daniels
# 2021/7/30
#
# Analyze data from runLandauTestSimulations.py
#

import pandas as pd
import numpy as np
from toolbox.simplePickle import load,save
import glob

def trimFittingData(datafilePrefix):
    """
    Remove gaussian mixture fitting objects to reduce file size
    """
    # loop over all files with the given prefix
    fileList = glob.glob("{}*".format(datafilePrefix))
    if len(fileList) == 0:
        raise Exception("No files found with prefix {}".format(datafilePrefix))
    for file in fileList:
        d = load(file)
        for mu in d.keys():
            for gname in ['gaussianMixtureAnalysis1','gaussianMixtureAnalysisNone']:
                if gname in d[mu]:
                    d[mu][gname].pop('gSingle')
                    d[mu][gname].pop('gMultiple')
        newfilename = file[:-4]+'_trimmed.dat'
        print("trimFittingData: Saved data to {}".format(newfilename))
        save(d,newfilename)

def fittingData(datafilePrefix):

    dataDict = {}
    dfDict = {}
    
    # loop over all files with the given prefix
    fileList = glob.glob("{}*".format(datafilePrefix))
    if len(fileList) == 0:
        raise Exception("No files found with prefix {}".format(datafilePrefix))
    for file in fileList:
        dataDictSingle,dfSingle = fittingData_singleRun(file)
        runIndex = dataDictSingle[list(dataDictSingle.keys())[0]]["runIndex"]
        
        dataDict[runIndex] = dataDictSingle
        dfDict[runIndex] = dfSingle
        
    df = pd.concat(dfDict,names=["runIndex"])
    df = df.sort_values(['mu','runIndex']).reset_index().drop(columns='level_1')
    
    return dataDict,df

def fittingData_singleRun(datafile):
    
    dataDict = load(datafile)
    
    muList = dataDict.keys()
    
    minIndexList = []
    minValList = []
    bistableIndexList = []
    bistableLikelihoodList = []
    bistableBICList = []
    bistableCList = []
    bistableDList = []
    bistableNuMuList = []
    bistableEigvalList = []
    propAboveMeanList = []
    landauTimeList = []
    gaussianLLListNone = []
    gaussianBICListNone = []
    gaussianTimeMinutesListNone = []
    gaussianLLList1 = []
    gaussianBICList1 = []
    gaussianTimeMinutesList1 = []
    for mu in muList:
        if ('landauAnalysis' in dataDict[mu]) \
            and not np.isnan(dataDict[mu]['landauAnalysis']['llList'][0]):
            landauData = dataDict[mu]['landauAnalysis']
            
            # filter out any zero eigenvalues
            nonzeroEigs = np.where(landauData['valList'] != 0.)[0]
            dataDict[mu]['valList'] = landauData['valList'][nonzeroEigs]
            dataDict[mu]['vecList'] = np.array(landauData['vecList'])[nonzeroEigs]
           
            # find dimension with minimum eigenvalue (max variance)
            minIndex = np.argmin(landauData['valList'])
            minIndexList.append(minIndex)
            minValList.append(landauData['valList'][minIndex])
           
            # find dimension with most evidence for bistability
            bistableIndex = np.argmin(landauData['llList'])
            
            # extract bistability parameters for max bistability dimension
            bistableIndexList.append(bistableIndex)
            bistableLikelihoodList.append(landauData['llList'][bistableIndex])
            bistableBICList.append(landauData['bicDiffList'][bistableIndex])
            bistableCList.append(landauData['cList'][bistableIndex])
            bistableDList.append(landauData['dList'][bistableIndex])
            bistableNuMuList.append(landauData['nuMuList'][bistableIndex])
            bistableEigvalList.append(landauData['valList'][bistableIndex])
            
            # calculate proportion of samples above the mean in the bistable dimension
            vec = landauData['vecList'][bistableIndex]
            x = np.dot(dataDict[mu]['finalStates']-landauData['sampleMean'],vec)
            propAboveMeanList.append( np.mean(x > 0) )
            
            landauTimeList.append(landauData['landauTimeMinutes'])
            
        else: # no landau analysis, or an error in the Mathematica code
            minIndexList.append(np.nan)
            minValList.append(np.nan)
            
            bistableIndexList.append(np.nan)
            bistableLikelihoodList.append(np.nan)
            bistableBICList.append(np.nan)
            bistableCList.append(np.nan)
            bistableDList.append(np.nan)
            bistableNuMuList.append(np.nan)
            bistableEigvalList.append(np.nan)
            
            propAboveMeanList.append(np.nan)
            
            landauTimeList.append(np.nan)
        
        if 'gaussianMixtureAnalysisNone' in dataDict[mu]:
            gNone = dataDict[mu]['gaussianMixtureAnalysisNone']
            gaussianLLListNone.append(gNone['llDiff'])
            gaussianBICListNone.append(gNone['bicDiff'])
            gaussianTimeMinutesListNone.append(gNone['gaussianTimeMinutes'])
        else:
            gaussianLLListNone.append(np.nan)
            gaussianBICListNone.append(np.nan)
            gaussianTimeMinutesListNone.append(np.nan)
            
        if 'gaussianMixtureAnalysis1' in dataDict[mu]:
            g1 = dataDict[mu]['gaussianMixtureAnalysis1']
            gaussianLLList1.append(g1['llDiff'])
            gaussianBICList1.append(g1['bicDiff'])
            gaussianTimeMinutesList1.append(g1['gaussianTimeMinutes'])
        else:
            gaussianLLList1.append(np.nan)
            gaussianBICList1.append(np.nan)
            gaussianTimeMinutesList1.append(np.nan)
        
    dfData = {'mu': muList,
              'bistable index': bistableIndexList,
              'bistable log-likelihood': bistableLikelihoodList,
              'bistable bic diff': bistableBICList,
              'gaussian unconstrained log-likelihood': gaussianLLListNone,
              'gaussian unconstrained bic diff': gaussianBICListNone,
              'gaussian PC log-likelihood': gaussianLLList1,
              'gaussian PC bic diff': gaussianBICList1,
              'bistable c': bistableCList,
              'bistable d': bistableDList,
              'bistable nuMu': bistableNuMuList,
              'bistable eigenvalue': bistableEigvalList,
              'prop. above mean in bistable dim.': propAboveMeanList,
              'min eigenvalue index': minIndexList,
              'min eigenvalue': minValList,
              'network name': [dataDict[mu]['networkName'] for mu in muList],
              'Ncomponents': [dataDict[mu]['Ncomponents'] for mu in muList],
              'Nsamples': [dataDict[mu]['Nsamples'] for mu in muList],
              'tFinal': [dataDict[mu]['tFinal'] for mu in muList],
              'landau time (m)':  landauTimeList,
              'gaussian unconstrained time (m)': gaussianTimeMinutesListNone,
              'gaussian PC time (m)': gaussianTimeMinutesList1,
              'initial seed': [dataDict[mu]['seedList'][0] for mu in muList],
              }
    
    df = pd.DataFrame.from_dict(dfData)
    
    return dataDict,df

def runtime(df):
    dftime = df[['runIndex',
                 'simulation time (m)',
                 'landau time (m)']].set_index('runIndex').sum(level='runIndex')
    dftime['total (m)'] = dftime['simulation time (m)'] + dftime['landau time (m)']
    return dftime
