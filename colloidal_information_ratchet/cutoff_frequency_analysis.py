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
SAVE_DATA = 1

F_NE_MIN_POWER = -3
F_NE_MAX_POWER = 4
NUMBER_OF_POINTS = 30
D_NE = 3
DELTA_G = 0.37
    

def main(delta_g=DELTA_G, d_ne=D_NE, model=constants.NOISE_MODEL):
    
    f_ne_logspace = np.logspace(F_NE_MIN_POWER, F_NE_MAX_POWER,
                                num=NUMBER_OF_POINTS, base=10)
    
    p = Pool()
    args = [(f_ne, d_ne, delta_g) for f_ne in f_ne_logspace]
    result_array  = p.starmap(calculate_mean_free_power, args)
    result_array = np.array(result_array)
    mean_free_powers = result_array[:,0]
    std_free_powers = result_array[:,1]
        
    if SAVE_DATA:
        
        file = np.zeros((len(f_ne_logspace), 3))
        file[:,0] = f_ne_logspace
        file[:,1] = mean_free_powers
        file[:,2] = std_free_powers
        headers = ["cutoff_frequency", "mean_free_power", "std_free_power"]
        df = pd.DataFrame(file, columns=headers)
        df.to_csv((DATA_DIRECTORY + 
                   "power_against_cutoff_frequency_error_bars_model_{}.csv".format(
                       model)),
                  index=0, header=1, sep=',')
    
  
if __name__ == '__main__':
    main()
