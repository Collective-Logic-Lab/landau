# runLandauTestSimulations.py
#
# Bryan Daniels
# 2021/7/30
#
# Run multiple collective dynamics test simulations and Landau analysis
# and save the results.
#

from criticalDynamics import simpleCollectiveDynamics,allToAllNetworkAdjacency
from landauAnalysis import landauAnalysis
import time
import numpy as np
import pandas as pd
import subprocess # for getGitHash
from toolbox.simplePickle import save


def runMultipleMus(mus,originalWeightMatrix,baseDict={},
    Nsamples=100,tFinal=100,seedStart=123,verbose=True):
    """
    Run sampling of final states for multiple interaction strengths mu
    that multiply values of the network specified by originalWeightMatrix.
    
    Returns dataDict indexed by mu.
    """

    dataDict = {}
    for mu in mus:
        dataDict[mu] = baseDict

    # loop over mu (interaction strength)
    for muIndex,mu in enumerate(mus):
        
        weightMatrix = mu * originalWeightMatrix
        
        finalStates = []
        seedList = []
        startTime = time.time()
        
        # loop over samples with fixed weightMatrix
        for i in range(Nsamples):
            seed = seedStart + 10*len(mus)*i + muIndex
            seedList.append(seed)

            # run single simulation to tFinal and save final state
            timeSeriesData = simpleCollectiveDynamics(weightMatrix,
                                                      tFinal=tFinal,
                                                      seed=seed)
            finalState = timeSeriesData.loc[tFinal]
            finalStates.append(finalState)

        finalStates = pd.DataFrame(finalStates)
        simTimeMinutes = (time.time() - startTime)/60.
        
        # run Landau analysis on samples of final states
        startTime = time.time()
        sampleMean,valList,vecList,llList,cList,dList = landauAnalysis(finalStates)
        landauTimeMinutes = (time.time() - startTime)/60.
        
        dataDict[mu].update( {'mu': mu,
                        'originalWeightMatrix': originalWeightMatrix,
                        'Nsamples': Nsamples,
                        'tFinal': tFinal,
                        'seedList': seedList,
                        'finalStates': finalStates,
                        'simTimeMinutes': simTimeMinutes,
                        'landauTimeMinutes': landauTimeMinutes,
                        'sampleMean': sampleMean,
                        'valList': valList,
                        'vecList': vecList,
                        'llList': llList,
                        'cList': cList,
                        'dList': dList,
                       } )
        
        if verbose:
            print("runMultipleMus: Done with mu = {}".format(mu))
                       
    return dataDict
        
def getGitHash(dir='./'):
    """
    Get the hash code for the current HEAD state of the git repository
    in directory 'dir'.
    """
    hashVal = subprocess.check_output(['git','rev-parse','HEAD'],cwd=dir)
    return hashVal.strip().decode()

if __name__ == '__main__':

    # set up parameters of run
    Ncomponents = 10 #50 #100 #10
    Nsamples = 100 #100
    tFinal = 100
    networkName = 'allToAll'
    muMin,muMax = 0./Ncomponents,2./Ncomponents
    Nmus = 101 #11 #101
    seedStart = 123
        
    mus = np.linspace(muMin,muMax,Nmus)
    if networkName == 'allToAll':
        weightMatrix = allToAllNetworkAdjacency(Ncomponents)
    else:
        raise(Exception)

    baseDict = {'networkName': networkName,
                'Ncomponents': Ncomponents,
                'gitHash': getGitHash(),
               }

    # run the simulations and analysis
    dataDict = runMultipleMus(mus,
                              weightMatrix,
                              baseDict=baseDict,
                              Nsamples=Nsamples,
                              tFinal=tFinal,
                              seedStart=seedStart)

    # save data
    filename = 'LandauTestData_{}_Ncomponents{}_Nmus{}.dat'.format(
                networkName,Ncomponents,Nmus)
    save(dataDict,filename)
    print("runLandauTestSimulations: Saved data to {}".format(filename))




        
        
