# -*- coding: utf-8 -*-
# cython: language_level=3
"""
Noise model 1: Active Ornstein-Uhlenbeck noise - noise term inversely
proportional to noise correlation time

Noise model 2: Power limited noise - noise term inversely proportional to
square root of noise correlation time

@author: Lewis Dean
"""

NOISE_MODEL = 1
EQUILIBRIUM_NOISE_INITIALISATION = 1
EQUILIBRIUM_X = 1

SAMPLE_PATHS = 1000
TRAJECTORY_STEPS = 10**6
TRAJECTORY_TIME = 100
DELTA_T = TRAJECTORY_TIME / TRAJECTORY_STEPS

MEASURING_FREQUENCY = 1/DELTA_T # Continuous sampling limit
THRESHOLD = 0
OFFSET = 0
