# -*- coding: utf-8 -*-
# cython: language_level=3
"""
Created on Sun Jul  9 16:59:05 2023

test...

@author: lewis
"""

from ensemble_generator_cython import generate_trajectory_ensemble
import active_bath_constants_cython as constants

import matplotlib.pyplot as plt
import numpy as np
cimport numpy as cnp


cdef str IMAGE_DIRECTORY = "plots/"
cdef bint PLOT_FREE_POWER = 0
cdef bint PLOT_SAVE = 1


cnp.import_array()
doubles = np.float64
longlongs = np.longlong
ctypedef cnp.double_t doubles_t
ctypedef cnp.longlong_t longlongs_t

cpdef return_empty(shape):
    
    return np.empty(shape, doubles)


cpdef delta_f(double lmb_n0, double lmb_n1, float delta_g):
    
    return delta_g*(lmb_n1 - lmb_n0)


cpdef cnp.ndarray[cnp.double_t, ndim=1] generate_free_power_ensemble(
    cnp.ndarray[cnp.double_t, ndim=2] well_positions,
    float delta_g,
    float total_time,
    int samples=constants.SAMPLE_PATHS,
    int steps=constants.N,
    float transient_fraction=constants.TRANSIENT_FRACTION):
    
    
    cdef int index, count
    cdef double lmb
    cdef cnp.ndarray[cnp.double_t, ndim=1] path, no_transience_path
    cdef int transient_steps = round(steps*transient_fraction)
    cdef float sampling_time = total_time*(1-transient_fraction)
    
    cdef cnp.ndarray[cnp.double_t, ndim=2] ensemble_free_powers = (
        return_empty((samples, steps-transient_steps + 1)))
    
    for index, path in enumerate(well_positions):
        no_transience_path = path[transient_steps:]
        for count, lmb in enumerate(no_transience_path):
            if count < (steps - transient_steps):  
        
                ensemble_free_powers[index, count] = (
                    delta_f(lmb, no_transience_path[count+1], delta_g))
        
        average_free_powers = 1/(sampling_time) * np.sum(ensemble_free_powers, axis=1)
        
    return average_free_powers


cpdef plot_free_power(cnp.ndarray[cnp.double_t, ndim=1] free_energies,
                      float f_ne,
                      float d_ne,
                      float delta_g,
                      float total_time=constants.PROTOCOL_TIME,
                      float transient_fraction=constants.TRANSIENT_FRACTION,
                      int model=constants.NOISE_MODEL):
        
        cdef float sampling_time = total_time*(1-transient_fraction)
        cdef cnp.ndarray[cnp.longlong_t, ndim=1] fe_counts
        cdef cnp.ndarray[cnp.double_t, ndim=1] fe_bins
    
        fe_counts, fe_bins = np.histogram(free_energies, bins=50)
        
        plt.title("Distribution of {} ".format(sampling_time) +
                  r"$\tau_r$ average free power outputs")
        plt.xlabel("Average free power output")
        plt.ylabel("Counts")
        
        plt.stairs(fe_counts, fe_bins, fill=True,
                   label=(r"$f_{ne}$ =" +" {:.3g}".format(f_ne) + "\n" +
                          r"$D_{ne}$ =" + " {:.3g}".format(d_ne) + "\n" +
                          r"$\delta_g$ =" + " {:.3g}".format(delta_g)))
        plt.legend()
        
        if PLOT_SAVE:
            
            if f_ne < 1:
                plt.savefig((IMAGE_DIRECTORY + 
                             "low_f_ne_free_power-distribution_model_{}.png".format(
                                 model)), dpi=600)
            else:
                plt.savefig((IMAGE_DIRECTORY + 
                             "high_f_ne_free_power-distribution_model_{}.png".format(
                                 model)), dpi=600)
                
        
        plt.show()
        plt.close()
        
        
cpdef cnp.ndarray[cnp.double_t, ndim=1] calculate_mean_free_power(
    float f_ne, float d_ne, float delta_g,
    float time=constants.PROTOCOL_TIME):
    '''
    ddof=1 in np.std() uses the Bessel correction for sample
    variance.
    '''
    
    cdef cnp.ndarray[cnp.double_t, ndim=2] well_positions = (
    generate_trajectory_ensemble(f_ne, d_ne, delta_g))
    
    cdef cnp.ndarray[cnp.double_t, ndim=1] free_energy_ensemble = (
    generate_free_power_ensemble(well_positions, delta_g, time))
    
    cdef cnp.ndarray[cnp.double_t, ndim=1] results = (
        return_empty(2))
    results[0] = np.mean(free_energy_ensemble)
    results[1] = np.std(free_energy_ensemble, ddof=1)

    if PLOT_FREE_POWER:
        
        plot_free_power(free_energy_ensemble, f_ne, d_ne, delta_g)
            
    return results


def gaussian_test(float f_ne, float d_ne, float delta_g,
                  float time=constants.PROTOCOL_TIME):
    
    cdef cnp.ndarray[cnp.double_t, ndim=2] well_positions = (
    generate_trajectory_ensemble(f_ne, d_ne, delta_g))
    
    cdef cnp.ndarray[cnp.double_t, ndim=1] free_energy_ensemble = (
    generate_free_power_ensemble(well_positions, delta_g, time))
    
    plot_free_power(free_energy_ensemble, f_ne, d_ne, delta_g)
    
    