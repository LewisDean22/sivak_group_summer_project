# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:52:20 2023

@author: lewis
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

import active_bath_constants_cython as constants
from free_power_calculator_cython import calculate_mean_free_power

DATA_DIRECTORY = "data_files/"
NUMBER_OF_POINTS = 30
SAVE_DATA = 1

D_NE_MIN_POWER = -1
D_NE_MAX_POWER = 2
TAU_C = 1 / 118
DELTA_G = 0.38
    

def main(delta_g=DELTA_G, tau_c=TAU_C, model=constants.NOISE_MODEL):

    
    d_ne_logspace = np.logspace(D_NE_MIN_POWER, D_NE_MAX_POWER,
                                num=NUMBER_OF_POINTS, base=10)
                                  
    p = Pool()
    arguments = [(tau_c, d_ne, delta_g) for d_ne in d_ne_logspace] 
    result_array = p.starmap(calculate_mean_free_power, arguments)
    result_array = np.array(result_array)
    mean_free_powers = result_array[:,0]
    std_mean_free_powers = result_array[:,1]
        
    if SAVE_DATA:
        
        file = np.zeros((len(d_ne_logspace), 3))
        file[:,0] = d_ne_logspace
        file[:,1] = mean_free_powers
        file[:,2] = std_mean_free_powers
        headers = ["noise_strength", "mean_free_power", "std_mean_free_power"]
        df = pd.DataFrame(file, columns=headers)
        df.to_csv((DATA_DIRECTORY +
                  "power_against_noise_strength_model_{}.csv".format(
                      model)),
                  index=0, header=1, sep=',')
    
  
if __name__ == '__main__':
    main()
