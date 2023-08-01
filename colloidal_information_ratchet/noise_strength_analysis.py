# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:52:20 2023

@author: lewis
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

from free_power_calculator_cython import calculate_mean_free_power

DATA_DIRECTORY = "data_files/"

NUMBER_OF_POINTS = 30
SAVE_DATA = 1

D_NE_MIN_POWER = -1
D_NE_MAX_POWER = 2
F_NE = 118
DELTA_G = 0.38
    

def main(delta_g=DELTA_G, f_ne=F_NE):

    
    d_ne_logspace = np.logspace(D_NE_MIN_POWER, D_NE_MAX_POWER,
                                num=NUMBER_OF_POINTS, base=10)
                                  
    p = Pool()
    arguments = [(f_ne, d_ne, delta_g) for d_ne in d_ne_logspace] 
    result_array = p.starmap(calculate_mean_free_power, arguments)
    result_array = np.array(result_array)
    mean_free_powers = result_array[:,0]
    std_free_powers = result_array[:,1]
        
    if SAVE_DATA:
        
        file = np.zeros((len(d_ne_logspace), 3))
        file[:,0] = d_ne_logspace
        file[:,1] = mean_free_powers
        file[:,2] = std_free_powers
        headers = ["noise_strength", "mean_free_power", "std_free_power"]
        df = pd.DataFrame(file, columns=headers)
        df.to_csv(DATA_DIRECTORY + "power_against_noise_strength_error_bars.csv",
                  index=0, header=1, sep=',')
    
  
if __name__ == '__main__':
    main()
