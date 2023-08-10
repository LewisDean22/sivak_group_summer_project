# -*- coding: utf-8 -*-
# cython: language_level=3
"""
Created on Tue Jul 4 18:57:42 2023

Does this process involve a stiff SDE? Is there disparate
time scales for relaxation to thermal equilibrium and
the time between feedback updates?

Only free energy is a useful power output here as the negative
trap work is dissipated as heat to the colloidal medium. Hence,
the zero work feedback scheme will be used here, with alpha=2.

@author: Lewis Dean
"""

import numpy as np
cimport numpy as cnp
from scipy.constants import g, k #noqa
import active_bath_constants_cython as constants


cnp.import_array()
doubles = np.float64
ctypedef cnp.double_t doubles_t


cpdef return_empty(shape):
    
    return np.empty(shape, doubles)


cpdef double drift_term(double x, double lmb, float delta_g):
    
    return -(x-lmb) - delta_g


cpdef double equilibrium_diffusion_term(double dt):
    
    return np.sqrt(2*dt)


cpdef double noise_drift_term(double zeta, float tau_c):
    
    return -zeta / tau_c
    

cpdef double noise_diffusion_term_1(float tau_c, float d_ne, double dt):
    
    return np.sqrt(2*d_ne*dt) / tau_c


cpdef double noise_diffusion_term_2(float tau_c, float d_ne, double dt):
    
    return np.sqrt(2*d_ne*dt/tau_c)


cpdef float wiener_increment():
    
    return np.random.normal(0, 1)


cpdef double delta_zeta(double zeta, float tau_c, float d_ne,
                        double dt, double dw,
                        int model=constants.NOISE_MODEL):
    
    if model == 1:
        
        return noise_drift_term(zeta, tau_c)*dt + noise_diffusion_term_1(
            tau_c, d_ne, dt)*dw
    
    if model == 2:
        
        return noise_drift_term(zeta, tau_c)*dt + noise_diffusion_term_2(
            tau_c, d_ne, dt)*dw
    
    
cpdef double zeta_heun_method(double zeta, float tau_c, float d_ne, float delta_g,
                              double dt=constants.DELTA_T,
                              int model=constants.NOISE_MODEL):
    
       cdef double dw_zeta, zeta_estimate
    
       dw_zeta = wiener_increment()
       zeta_estimate = zeta + delta_zeta(zeta, tau_c, d_ne, dt, dw_zeta)
       
       if model == 1:
           
           zeta = (zeta + 0.5*dt*(noise_drift_term(zeta, tau_c) + 
                   noise_drift_term(zeta_estimate, tau_c)) + 
                   noise_diffusion_term_1(tau_c, d_ne, dt)*dw_zeta)
           
           return zeta
           
       if model == 2:
            
           zeta = (zeta + 0.5*dt*(noise_drift_term(zeta, tau_c) + 
                   noise_drift_term(zeta_estimate, tau_c)) + 
                   noise_diffusion_term_2(tau_c, d_ne, dt)*dw_zeta)
            
           return zeta
     


cpdef double delta_x(double x, double lmb, float delta_g,
                     double zeta, double dt, double dw):
    
    return (drift_term(x, lmb, delta_g) + zeta)*dt + equilibrium_diffusion_term(dt) * dw


cpdef cnp.ndarray[cnp.double_t, ndim=1] x_euler_method(float tau_c, float d_ne, float delta_g,
                                                      int steps=constants.N,
                                                      double freq_measure=constants.MEASURING_FREQUENCY,
                                                      double dt=constants.DELTA_T,
                                                      float threshold=constants.THRESHOLD,
                                                      float gain=constants.GAIN,
                                                      float offset=constants.OFFSET):
    
    cdef int n
    cdef double time, x, zeta, lmb
    cdef cnp.ndarray[cnp.double_t, ndim=1] lmb_array = (
        return_empty(steps+1))
    
    time = 0
    x = - delta_g
    zeta = 0
    lmb_array[0] = 0
    
    for n in range(steps):
        lmb = lmb_array[n]
        
        zeta = zeta_heun_method(zeta, tau_c, d_ne, delta_g)
           
        dw_x = wiener_increment()
        x = x + delta_x(x, lmb, delta_g, zeta, dt, dw_x)
        
        # dw_x = wiener_increment()
        # x_estimate = x + delta_x(x, lmb, delta_g, zeta, dt, dw_x)
        # x = (x + 0.5*dt*(drift_term(x, lmb, delta_g) + 
        #         drift_term(x_estimate, lmb, delta_g)) + 
        #         equilibrium_diffusion_term(dt)*dw_x)
        
        time = (n+1)*dt
        
        if ((time*freq_measure).is_integer() and 
            (x - lmb) > threshold):
            lmb_array[n+1] = lmb + gain*(x - lmb) + offset
        else:
            lmb_array[n+1] = lmb
        
    
    return lmb_array


cpdef cnp.ndarray[cnp.double_t, ndim=2] generate_trajectory_ensemble(float tau_c, float d_ne, float delta_g,
                                                                     int steps=constants.N,
                                                                     int samples=constants.SAMPLE_PATHS):
   
    cdef int j
    cdef cnp.ndarray[cnp.double_t, ndim=2] trajectories = (
        return_empty((samples, steps+1)))
    
    for j in range(samples):
        trajectories[j] = x_euler_method(tau_c, d_ne, delta_g)
    
    return trajectories
