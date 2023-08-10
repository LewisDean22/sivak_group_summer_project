# -*- coding: utf-8 -*-
# cython: language_level=3
"""
Noise model 1: Coloured Ornstein-Uhlenbeck noise - noise term inversely
proprtional to noise correlation time

Noise model 2: Power limited noise - noise term inversely proprtional to
square root of noise correlation time

@author: Lewis Dean
"""

NOISE_MODEL = 1

SAMPLE_PATHS = 1000
N = 100_000 # Number of time steps
PROTOCOL_TIME = 10
DELTA_T = PROTOCOL_TIME / N

MEASURING_FREQUENCY = 1/DELTA_T # Continuous sampling limit (all discrete timesteps)
THRESHOLD = 0
OFFSET = 0

TRANSIENT_FRACTION = 0.1