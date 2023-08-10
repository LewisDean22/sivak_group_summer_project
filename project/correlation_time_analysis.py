# -*- coding: utf-8 -*-
""" @author: Lewis Dean """

import numpy as np
import pandas as pd
from multiprocessing import Pool

import active_bath_constants_cython as constants
from free_power_calculator_cython import calculate_mean_free_power

DATA_DIRECTORY = "data_files/"
SAVE_DATA = 1
NUMBER_OF_POINTS = 30

TAU_C_MIN_POWER = -4
TAU_C_MAX_POWER = 3
D_NE = 3
DELTA_G = 0.37
    

def main(delta_g=DELTA_G, d_ne=D_NE, model=constants.NOISE_MODEL):
    
    tau_c_logspace = np.logspace(TAU_C_MIN_POWER, TAU_C_MAX_POWER,
                                num=NUMBER_OF_POINTS, base=10)
    
    p = Pool()
    args = [(tau_c, d_ne, delta_g) for tau_c in tau_c_logspace]
    result_array  = p.starmap(calculate_mean_free_power, args)
    result_array = np.array(result_array)
    mean_free_powers = result_array[:,0]
    std_mean_free_powers = result_array[:,1]
        
    if SAVE_DATA:
        
        file = np.zeros((len(tau_c_logspace), 3))
        file[:,0] = tau_c_logspace
        file[:,1] = mean_free_powers
        file[:,2] = std_mean_free_powers
        headers = ["correlation_time", "mean_free_power", "std_mean_free_power"]
        df = pd.DataFrame(file, columns=headers)
        df.to_csv((DATA_DIRECTORY + 
                   "power_against_correlation_time_model_{}.csv".format(
                       model)),
                  index=0, header=1, sep=',')
    
  
if __name__ == '__main__':
    main()
