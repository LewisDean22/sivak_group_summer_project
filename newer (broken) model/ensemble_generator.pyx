# -*- coding: utf-8 -*-
# cython: language_level=3
"""
ISSUES: average power calculations are way off and sometimes the trap work
gives a nan value. Suggests maybe zero multiplied by some very large number.
x quite easily explodes.

@author: Lewis Dean
"""

import numpy as np
cimport numpy as cnp
from scipy.stats import bootstrap
import model_constants as constants

cnp.import_array()
doubles = np.float64
ctypedef cnp.double_t doubles_t


cpdef return_empty(shape):
    return np.empty(shape, doubles)


cpdef double drift_term(double x, double lmb, float delta_g):
    
    return -(x-lmb) - delta_g


cpdef double equilibrium_diffusion_term(double dt=constants.DELTA_T):
    
    return np.sqrt(2*dt)


cpdef double noise_drift_term(double zeta, float tau_c):
    
    return -zeta / tau_c


cpdef double noise_diffusion_term(float tau_c, float d_ne,
                                  double dt=constants.DELTA_T,
                                  int model=constants.NOISE_MODEL):
    
    if model == 1:
        return np.sqrt(2*d_ne*dt) / tau_c
    elif model == 2:
        return np.sqrt(2*d_ne*dt/tau_c)


cpdef float wiener_increment():
    
    return np.random.normal(0, 1)


cpdef float generate_initial_active_noise(
    float tau_c, float d_ne,
    int model=constants.NOISE_MODEL,
    int equilibrium_noise=constants.EQUILIBRIUM_NOISE_INITIALISATION):
    
    if not equilibrium_noise:
        return 0
    
    cdef float noise_std, initial_noise

    if model == 1:
        noise_std = np.sqrt(d_ne / tau_c)
    elif model == 2:
        noise_std = np.sqrt(d_ne)
    
    initial_noise = np.random.normal(0, noise_std)
    
    return initial_noise


cpdef float generate_initial_x(
    float tau_c, float d_ne, float delta_g,
    int model=constants.NOISE_MODEL,
    int equilibrium_x=constants.EQUILIBRIUM_X):
    
    if not equilibrium_x:
        return -delta_g
    
    cdef float x_std, initial_x

    if model == 1:
        x_std = np.sqrt(1 + d_ne/(1 + tau_c))
    elif model == 2:
        x_std = np.sqrt(1 + d_ne*tau_c/(1 + tau_c))
    
    initial_x = np.random.normal(-delta_g, x_std)
    
    return initial_x


cpdef double delta_zeta(double zeta, float tau_c, float d_ne, double dw,
                        double dt=constants.DELTA_T):

    return noise_drift_term(zeta, tau_c)*dt + noise_diffusion_term(
        tau_c, d_ne)*dw
    

cpdef double zeta_heun_method(double zeta, float tau_c, float d_ne,
                              double dt=constants.DELTA_T,
                              int model=constants.NOISE_MODEL):
 
    cdef double dw_zeta, zeta_estimate
 
    dw_zeta = wiener_increment()
    zeta_estimate = zeta + delta_zeta(zeta, tau_c, d_ne, dw_zeta)
         
    zeta += (0.5*dt*(noise_drift_term(zeta, tau_c) + 
            noise_drift_term(zeta_estimate, tau_c)) + 
            noise_diffusion_term(tau_c, d_ne)*dw_zeta)
         
    return zeta
     

cpdef double delta_x(double x, double lmb, float delta_g,
                     double zeta, double dw_x, double dt=constants.DELTA_T):
    
    return (drift_term(x, lmb, delta_g) + zeta)*dt + equilibrium_diffusion_term() * dw_x


cpdef x_euler_method(double x, double lmb, double zeta, double time,
                     float tau_c, float d_ne, float gain,
                     float delta_g, double dt=constants.DELTA_T):

    cdef double dw_x

    zeta = zeta_heun_method(zeta, tau_c, d_ne)
    dw_x = wiener_increment()
    x += delta_x(x, lmb, delta_g, zeta, dw_x)
    
    time += dt
                
    return x, time


cpdef double calculate_trap_work(double x, double lmb, float gain):

    return 0.5*gain*(gain-2)*(x - lmb)**2


cpdef cnp.ndarray[cnp.double_t, ndim=1] power_calculator(
    double x_0, double lmb_0, double zeta_0,
    float tau_c, float d_ne, float gain,
    float delta_g, int steps=constants.TRAJECTORY_STEPS,
    float trajectory_time=constants.TRAJECTORY_TIME,
    double measure_freq=constants.MEASURING_FREQUENCY,
    double dt=constants.DELTA_T,
    float threshold=constants.THRESHOLD,
    float offset=constants.OFFSET):
    '''
    This new function returns average free power and trap power along
    the trajectory. Note that trajecory time is now some new parameter
    which is the supertrajectory time divided by the number of samples+1.
    
    '''
    
    cdef int n
    cdef double time, zeta, x, lmb, total_trap_work
    
    time = 0
    total_trap_work = 0
    x = x_0
    lmb = lmb_0
    zeta = zeta_0
    
    for n in range(steps):
        
        x, time = x_euler_method(x, lmb, zeta, time, tau_c,
                                 d_ne, gain, delta_g)

        if ((time*measure_freq).is_integer() and 
            (x - lmb) > threshold):
            total_trap_work += calculate_trap_work(x, lmb, gain)
            lmb += gain*(x - lmb) + offset
    
    free_power = delta_g*(lmb-lmb_0)/trajectory_time
    trap_power = total_trap_work/trajectory_time

    return np.array([free_power, trap_power, x, lmb, zeta])


cpdef double standard_error_on_mean(
    cnp.ndarray[cnp.double_t, ndim=1] ensemble):
   
    cdef double i 
    cdef bint all_zeros = 1
    
    for i in ensemble:
        if i != 0:
            all_zeros = 0
            break
    if all_zeros:
        return 0
    
    cdef cnp.ndarray[cnp.double_t, ndim=2] ensemble_samples = (
        np.reshape(ensemble, (1, len(ensemble))))

    mean_distribution = bootstrap(ensemble_samples, np.mean,
                                  n_resamples=1000)
    
    return mean_distribution.standard_error


cpdef cnp.ndarray[cnp.double_t, ndim=1] calculate_mean_powers(
    float tau_c, float d_ne, float gain,
    float delta_g, int samples=constants.SAMPLE_PATHS):
   
    cdef int j
    cdef bint burn_in = 1
    cdef double x, lmb, zeta
    cdef cnp.ndarray[cnp.double_t, ndim=1] free_power_array = np.array([])
    cdef cnp.ndarray[cnp.double_t, ndim=1] trap_power_array = np.array([])

    x = generate_initial_x(tau_c, d_ne, delta_g)
    lmb = 0
    zeta = generate_initial_active_noise(tau_c, d_ne)
    
    for j in range(samples):
        
        free_power, trap_power, x, lmb, zeta = power_calculator(x, lmb, zeta,
                                                                tau_c, d_ne,
                                                                gain, delta_g)
        if burn_in:
            burn_in = 0
        else:
            free_power_array = np.append(free_power_array, free_power)
            trap_power_array = np.append(trap_power_array, trap_power)
    
    cdef cnp.ndarray[cnp.double_t, ndim=1] mean_power_results = (
        return_empty(4))
    mean_power_results[0] = np.mean(free_power_array)
    mean_power_results[1] = standard_error_on_mean(free_power_array)
    mean_power_results[2] = np.mean(trap_power_array)
    mean_power_results[3] = standard_error_on_mean(trap_power_array)
    
    return mean_power_results
