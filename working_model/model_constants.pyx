# -*- coding: utf-8 -*-
# cython: language_level=3
"""
Noise model 1: Active Ornstein-Uhlenbeck noise (AOUP) - noise term
inversely proportional to noise correlation time. Power spectral density
inversely proportional to this correlation time.

Noise model 2: Power limited Ornstein-Uhlenbeck noise (PLOUP) - noise term
inversely proportional to square root of noise correlation time. Power
spectral density independent of this correlation time.

@author: Lewis Dean - lewis.dean@manchester.student.ac.uk
"""

NOISE_MODEL = 2
EQUILIBRIUM_NOISE_INITIALISATION = 1
EQUILIBRIUM_X = 1

SAMPLE_PATHS = 100
N = 1_000_000 # Number of time steps
PROTOCOL_TIME = 10**4
DELTA_T = PROTOCOL_TIME / N

MEASURING_FREQUENCY = 1/DELTA_T # Continuous sampling limit
THRESHOLD = 0
OFFSET = 0

TRANSIENT_FRACTION = 0.4