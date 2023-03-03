# criticalDynamics.py
#
# Bryan Daniels
# 2021/7/29
#
# Implement simple Langevin dynamical simulation that has a
# critical transition.
#

import numpy as np
import pandas as pd

from numba import jit

def simpleCollectiveDynamics(weightMatrix,inputConst=0,noiseVar=1,tFinal=10,
    deltat=1e-3,initialState=None,seed=None,onlyReturnFinalState=False):
    """
    Simulates the following stochastic process:
    
    dx_i / dt = inputConst - x_i + sum_j weightMatrix_{i,j} tanh(x_j) + xi
    
    where xi is uncorrelated Gaussian noise with variance 'noiseVar' per unit time.
    
    Time is discretized into units of deltat, and the simulation is run until time tFinal.
    
    weightMatrix                      : (N x N) matrix indicating the strength with
                                        which component j influences component i
    initialState (None)               : If given a list of length N, start the system in the
                                        given state.  If None, initial state defaults to
                                        all zeros.
    onlyReturnFinalState (False)      : If True, only the final state is returned
    """

    N = len(weightMatrix)
    
    if onlyReturnFinalState:
        times = [tFinal]
    else:
        times = np.arange(0,tFinal+deltat,deltat)

    # use numba function for speed
    stateList = simpleCollectiveDynamics_numba(weightMatrix,
                    inputConst=inputConst,
                    noiseVar=noiseVar,
                    tFinal=tFinal,
                    deltat=deltat,
                    initialState=initialState,
                    seed=seed,
                    onlyReturnFinalState=onlyReturnFinalState)
    
    # return simulation output as a pandas dataframe
    df = pd.DataFrame(stateList,
                      index=times,
                      columns=['Component {}'.format(i) for i in range(N)])
    df.index.set_names('Time',inplace=True)
    return df

@jit(nopython=True)
def simpleCollectiveDynamics_numba(weightMatrix,inputConst=0,noiseVar=1,tFinal=10,
    deltat=1e-3,initialState=None,seed=None,onlyReturnFinalState=False):
    """
    Simulates the following stochastic process:
    
    dx_i / dt = inputConst - x_i + sum_j weightMatrix_{i,j} tanh(x_j) + xi
    
    where xi is uncorrelated Gaussian noise with variance 'noiseVar' per unit time.
    
    Time is discretized into units of deltat, and the simulation is run until time tFinal.
    
    weightMatrix                      : (N x N) matrix indicating the strength with
                                        which component j influences component i
    initialState (None)               : If given a list of length N, start the system in the
                                        given state.  If None, initial state defaults to
                                        all zeros.
    onlyReturnFinalState (False)      : If True, only the final state is returned
    """
    np.random.seed(seed)
    
    N = len(weightMatrix)
    # make sure the weight matrix is square
    assert(len(weightMatrix[0])==N)
    
    # set up the initial state
    if initialState is None:
        state0 = np.zeros(N)
    else:
        state0 = initialState
    # make sure the initial state has the correct length
    assert(len(state0)==N)
    
    # set up the simulation times
    times = np.arange(0,tFinal+deltat,deltat)
    # make a list to hold the simulated steps
    if onlyReturnFinalState:
        stateList = np.empty((1,N))
    else:
        stateList = np.empty((len(times),N))
        stateList[0] = state0
    newState = state0
    
    # run the simulation (we already have the state for t=0)
    for i,time in enumerate(times[1:]):
        currentState = newState.copy()
        
        # compute deltax for current timestep
        deterministicPart = deltat*( inputConst - currentState + np.dot(weightMatrix,np.tanh(currentState)) )
        randomNumbers = np.array([np.random.normal() for i in range(N)])
        stochasticPart = np.sqrt(deltat*noiseVar)*randomNumbers
        deltax = deterministicPart + stochasticPart
        
        # update to find the new state
        newState = currentState + deltax
        
        if not onlyReturnFinalState:
            # record the new state
            stateList[i+1] = newState
       
    if onlyReturnFinalState:
        stateList[0] = newState
        
    return stateList

def allToAllNetworkAdjacency(N):
    return 1 - np.eye(N)
