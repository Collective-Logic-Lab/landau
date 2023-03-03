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
import sys
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
    
    Note that the lists valList, vecList, llList, cList, and dList
    are ordered such that the smallest indices correspond to largest variance.
    """

    dataDict = {}
    for mu in mus:
        dataDict[mu] = baseDict.copy()

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
                                                      seed=seed,
                                                      onlyReturnFinalState=True)
            finalState = timeSeriesData.loc[tFinal]
            finalStates.append(finalState)

        finalStates = pd.DataFrame(finalStates)
        simTimeMinutes = (time.time() - startTime)/60.
        
        dataDict[mu].update( {'mu': mu,
                        'originalWeightMatrix': originalWeightMatrix,
                        'Nsamples': Nsamples,
                        'tFinal': tFinal,
                        'seedList': seedList,
                        'finalStates': finalStates,
                        'simTimeMinutes': simTimeMinutes,
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
    Ncomponents = 91 #10 #50 #100 #10
    Nsamples = 1000 #16 #100
    tFinal = 100
    networkName = 'allToAll'
    muMin,muMax = 0./Ncomponents,2./Ncomponents
    Nmus = 51 #11 #101
    seedStart = 123
        
    # if command line argument is given, use it to modify seedStart
    if len(sys.argv) == 2:
        runIndex = int(sys.argv[1])
    elif len(sys.argv) > 2:
        print("Usage: python runLandauTestSimulations.py [runIndex]")
        exit()
    else:
        runIndex = 0
    seedStart += 1234567*runIndex

    mus = np.linspace(muMin,muMax,Nmus)
    if networkName == 'allToAll':
        weightMatrix = allToAllNetworkAdjacency(Ncomponents)
    else:
        raise(Exception)

    baseDict = {'networkName': networkName,
                'Ncomponents': Ncomponents,
                'gitHash': getGitHash(),
                'runIndex': runIndex,
                'runLandauAnalysis': runLandauAnalysis,
               }

    # run the simulations
    dataDict = runMultipleMus(mus,
                              weightMatrix,
                              baseDict=baseDict,
                              Nsamples=Nsamples,
                              tFinal=tFinal,
                              seedStart=seedStart)

    # save data
    filename = 'LandauTestData_{}_Ncomponents{}_Nsamples{}_Nmus{}_run{}.dat'.format(
                networkName,Ncomponents,Nsamples,Nmus,runIndex)
    save(dataDict,filename)
    print("runLandauTestSimulations: Saved data to {}".format(filename))




        
        
