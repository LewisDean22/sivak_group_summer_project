# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:40:20 2023

BROKEN - just fluctuates between 1 and 2

@author: lewis
"""

import numpy as np
import matplotlib.pyplot as plt
import time as tm

PLOT_TRAJECTORY = 1
PLOT_HISTOGRAM = 1

ITERATIONS = 100
N_0 = 2
DELTA_T = 10**-1

# Arrays organised as [[accessible states],  [transition_probabilities]]
# Divide latter by DELTA_T to get transition rate for that time step

# Cannot stay put in this markov chain
# Configuration drawn in my notepad
state_1 = [[1,2], [0.75/DELTA_T, 0.25/DELTA_T]]
state_2 = [[0,2,3], [0.375/DELTA_T, 0.375/DELTA_T, 0.25/DELTA_T]]
state_3 = [[0,1,3,4], [0.125/DELTA_T, 0.375/DELTA_T,
                       0.375/DELTA_T, 0.125/DELTA_T]]
state_4 = [[1,2,4], [0.25/DELTA_T, 0.375/DELTA_T, 0.375/DELTA_T]]
state_5 = [[2,3], [0.25/DELTA_T, 0.75/DELTA_T]]


def calculate_state_space_lambdas(state_space):
    
    state_space_lambdas = []
    
    for state in state_space:
        state_space_lambdas += [np.sum(state[1])]
    
    return state_space_lambdas


def get_tau(parameter):
    
    return np.random.exponential(1/parameter)
    
    
def find_new_state(state_space, state, parameter):
    
    r = np.random.uniform(0,1)
    n = 0
    cumulative_weight = 0

    while (state[1][n] + cumulative_weight)/parameter < r:
            cumulative_weight += state[1][n]
            n += 1
    
    end_state = state[0][n]
    return end_state
        
    
def main():
    
    state_space = [state_1, state_2, state_3, state_4, state_5]
    state_space_lambdas = calculate_state_space_lambdas(state_space)

    state_path = [N_0]
    time = 0
    
    for i in range(ITERATIONS):
    
        if PLOT_TRAJECTORY:
            fig = plt.figure(figsize=(10, 1))
            plt.title("t = {:.3f} state".format(time))
            plt.scatter(state_path[-1], 0, s=200)
            plt.xlim(0,4)
            plt.xlabel("State")
            plt.xticks([0,1,2,3,4])
            plt.ylim(-1,1)
            plt.show()
            plt.close()
            tm.sleep(0.1)
        
        state = state_space[state_path[-1]]
        parameter = state_space_lambdas[state_path[-1]]
        
        time += get_tau(parameter)
        state_path += [find_new_state(state_space, state, parameter)]

    if PLOT_HISTOGRAM:
        
        counts, bins = np.histogram(state_path, bins=[0,1,2,3,4])
        
        plt.stairs(counts, bins, fill=True)
        plt.title("Histogram of state occupancy")
        plt.xlabel("State")
        plt.xticks(bins)
        plt.ylabel("Count") 
        plt.show()
        plt.close()

if __name__ == '__main__':
    main()