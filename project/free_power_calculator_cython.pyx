# -*- coding: utf-8 -*-
# cython: language_level=3
"""
@author: Lewis Dean
"""

from ensemble_generator_cython import generate_trajectory_ensemble
import active_bath_constants_cython as constants

from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import numpy as np
cimport numpy as cnp


cdef str IMAGE_DIRECTORY = "plots/"
cdef bint PLOT_SAVE = 1


cnp.import_array()
doubles = np.float64
longlongs = np.longlong
ctypedef cnp.double_t doubles_t
ctypedef cnp.longlong_t longlongs_t

cpdef return_empty(shape):
    
    return np.empty(shape, doubles)


cpdef delta_free_energy(double lmb_n0, double lmb_n1, float delta_g):
    
    return delta_g*(lmb_n1 - lmb_n0)


cpdef cnp.ndarray[cnp.double_t, ndim=1] generate_free_power_ensemble(
    cnp.ndarray[cnp.double_t, ndim=2] trap_position_ensemble,
    float delta_g,
    float total_time,
    int samples=constants.SAMPLE_PATHS,
    int steps=constants.N,
    float transient_fraction=constants.TRANSIENT_FRACTION):
    
    
    cdef int sample_index, count
    cdef double lmb
    cdef cnp.ndarray[cnp.double_t, ndim=1] trap_positions, truncated_trap_positions
    cdef int transient_steps = round(steps*transient_fraction)
    cdef float sampling_time = total_time*(1-transient_fraction)
    
    cdef cnp.ndarray[cnp.double_t, ndim=2] work_ensemble = (
        return_empty((samples, steps-transient_steps + 1)))
    
    for sample_index, trap_positions in enumerate(trap_position_ensemble):
        truncated_trap_positions = trap_positions[transient_steps:]
        for count, lmb in enumerate(truncated_trap_positions):
            if count < (steps - transient_steps):  
        
                work_ensemble[sample_index, count] = delta_free_energy(
                    lmb, truncated_trap_positions[count+1], delta_g)
        
        free_power_ensemble = 1/(sampling_time) * np.sum(work_ensemble, axis=1)
        
    return free_power_ensemble


cpdef plot_free_power(cnp.ndarray[cnp.double_t, ndim=1] free_powers,
                      float tau_c,
                      float d_ne,
                      float delta_g,
                      float total_time=constants.PROTOCOL_TIME,
                      float transient_fraction=constants.TRANSIENT_FRACTION,
                      int model=constants.NOISE_MODEL):
        
        cdef float sampling_time = total_time*(1-transient_fraction)
        cdef cnp.ndarray[cnp.longlong_t, ndim=1] fe_counts
        cdef cnp.ndarray[cnp.double_t, ndim=1] fe_bins
    
        fe_counts, fe_bins = np.histogram(free_powers, bins=50)
        
        plt.title("Distribution of {} ".format(sampling_time) +
                  r"$\tau_r$ average free power outputs")
        plt.xlabel("Average free power output")
        plt.ylabel("Counts")
        
        plt.stairs(fe_counts, fe_bins, fill=True,
                   label=(r"$\tau_c$ =" +" {:.3g}".format(tau_c) + "\n" +
                          r"$D_{ne}$ =" + " {:.3g}".format(d_ne) + "\n" +
                          r"$\delta_g$ =" + " {:.3g}".format(delta_g)))
        plt.legend()
        
        if PLOT_SAVE:
            
            if tau_c < 1:
                plt.savefig((IMAGE_DIRECTORY + 
                             "low_tau_c_free_power-distribution_model_{}.png".format(
                                 model)), dpi=600)
            else:
                plt.savefig((IMAGE_DIRECTORY + 
                             "high_tau_c_free_power-distribution_model_{}.png".format(
                                 model)), dpi=600)
                
        
        plt.show()
        plt.close()


cpdef double standard_error_on_mean(
    cnp.ndarray[cnp.double_t, ndim=1] free_power_ensemble):
    '''
    ddof=1 in np.std() uses the Bessel correction for sample
    variance.
    '''
    
    cdef cnp.ndarray[cnp.double_t, ndim=2] free_power_ensemble_samples = (
        np.reshape(free_power_ensemble, (1, len(free_power_ensemble))))
    cdef cnp.ndarray[cnp.double_t, ndim=1] mean_free_power_ensemble
    
    res = bootstrap(free_power_ensemble_samples, np.mean, n_resamples=1000)
    
    return res.standard_error


cpdef cnp.ndarray[cnp.double_t, ndim=1] calculate_mean_free_power(
    float tau_c, float d_ne, float delta_g,
    float time=constants.PROTOCOL_TIME):
    
    cdef cnp.ndarray[cnp.double_t, ndim=2] trap_position_ensemble = (
    generate_trajectory_ensemble(tau_c, d_ne, delta_g))
    
    cdef cnp.ndarray[cnp.double_t, ndim=1] free_power_ensemble = (
    generate_free_power_ensemble(trap_position_ensemble, delta_g, time))
    
    cdef cnp.ndarray[cnp.double_t, ndim=1] results = (
        return_empty(2))
    results[0] = np.mean(free_power_ensemble)
    results[1] = standard_error_on_mean(free_power_ensemble)
            
    return results


cpdef gaussian_test(float tau_c, float d_ne, float delta_g,
                  float time=constants.PROTOCOL_TIME):
    
    cdef cnp.ndarray[cnp.double_t, ndim=2] trap_position_ensemble = (
    generate_trajectory_ensemble(tau_c, d_ne, delta_g))
    
    cdef cnp.ndarray[cnp.double_t, ndim=1] free_energy_ensemble = (
    generate_free_power_ensemble(trap_position_ensemble, delta_g, time))
    
    plot_free_power(free_energy_ensemble, tau_c, d_ne, delta_g)
    
    