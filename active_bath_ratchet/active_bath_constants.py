# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:06:03 2023

@author: lewis
"""

import numpy as np

SAMPLE_PATHS = 1
X_INIT = 0
LAMBDA_INIT = 0
N = 10_000 # Number of time steps
PROTOCOL_TIME = 10
DELTA_T = PROTOCOL_TIME / N

MEASURING_FREQUENCY = 1/DELTA_T
THRESHOLD = 0 # for optimal performance
OFFSET = 0
GAIN = 2
TRAP_CUTOFF_FREQUENCY = 1 / (2*np.pi) # set tau_r = 1 in equation

DELTA_G = 0.37



