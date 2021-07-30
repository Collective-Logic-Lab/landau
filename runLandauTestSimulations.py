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
from toolbox.simplePickle import save


def runMultipleMus(mus,originalWeightMatrix,Nsamples=100,tFinal=100,seedStart=123,
    verbose=True):
    """
    Run sampling of final states for multiple interaction strengths mu
    that multiply values of the network specified by originalWeightMatrix.
    
    Returns dataDict indexed by mu.
    """

    dataDict = {}

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
        
        dataDict[mu] = {'mu': mu,
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
                       }
        
        if verbose:
            print("runMultipleMus: Done with mu = {}".format(mu))
                       
    return dataDict
        

if __name__ == '__main__':

    # set up parameters of run
    Ncomponents = 10
    Nsamples = 10 #40 #100
    tFinal = 100
    networkName = 'allToAll'
    muMin,muMax = 0.3,0.7
    Nmus = 3
    seedStart = 123
        
    mus = np.linspace(muMin,muMax,Nmus)
    if networkName == 'allToAll':
        weightMatrix = allToAllNetworkAdjacency(Ncomponents)
    else:
        raise(Exception)

    # run the simulations and analysis
    dataDict = runMultipleMus(mus,
                              weightMatrix,
                              Nsamples=Nsamples,
                              tFinal=tFinal,
                              seedStart=seedStart)

    # save data
    for mu in dataDict.keys():
        dataDict[mu].update({'networkName': networkName,
                             'Ncomponents': Ncomponents,
                            })
    filename = 'LandauTestData_{}_Ncomponents{}_Nmus{}.dat'.format(
                networkName,Ncomponents,Nmus)
    save(dataDict,filename)
    print("runLandauTestSimulations: Saved data to {}".format(filename))




        
        
