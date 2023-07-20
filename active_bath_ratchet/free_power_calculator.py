# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:59:05 2023

@author: lewis
"""

from ensemble_generator import generate_trajectory_ensemble
import active_bath_constants as constants
import numpy as np
import matplotlib.pyplot as plt


IMAGE_DIRECTORY = "plots/"
PLOT_FREE_POWER = 0
PLOT_SAVE = 0


def delta_f(lmb_n0, lmb_n1, delta_g):
    
    return delta_g*(lmb_n1 - lmb_n0)


def generate_free_power_ensemble(well_positions, delta_g, total_time,
                                 samples=constants.SAMPLE_PATHS,
                                 n=constants.N):

    ensemble_free_power = np.empty(samples)
    
    for index, path in enumerate(well_positions):
        path_free_energy_array = []
        for count, lmb in enumerate(path):
            if count < n:  
        
                path_free_energy_array += [delta_f(lmb,
                                                   path[count+1],
                                                   delta_g)]
        
        average_free_power = 1/total_time * np.sum(path_free_energy_array)
        ensemble_free_power[index] = average_free_power
        
    return ensemble_free_power


def plot_free_power(free_energies,
                    total_time=constants.PROTOCOL_TIME):
    
        fe_counts, fe_bins = np.histogram(free_energies, bins=50)
        
        plt.title("Distribution of {} ".format(total_time) +
                  r"$\tau_r$ free power outputs")
        plt.xlabel("Free power")
        plt.ylabel("Count")
        
        plt.stairs(fe_counts, fe_bins, fill=True)
        
        if PLOT_SAVE:
            plt.savefig(IMAGE_DIRECTORY + "NE_engine_free_power-distribution.png",
                        dpi=600)
        
        plt.show()
        plt.close()
        
        
def calculate_mean_free_power(f_ne, d_ne, delta_g,
                              time=constants.PROTOCOL_TIME):

    well_positions = generate_trajectory_ensemble(f_ne, d_ne, delta_g)
    
    free_energy_ensemble = generate_free_power_ensemble(well_positions,
                                                        delta_g,
                                                        time)
    
    mean_free_power = np.mean(free_energy_ensemble)
                                  
    if PLOT_FREE_POWER:
        
        plot_free_power(free_energy_ensemble)
        
    return mean_free_power