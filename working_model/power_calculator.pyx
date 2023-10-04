# -*- coding: utf-8 -*-
# cython: language_level=3
"""
This script uses the ensemble_generator to produce the necessary samples of
the engine's operation and then calculates the mean free power associated with
the ensemble. Bootstrapping is also employed to find the standard error on this
mean. Again, the "broken_but_more_optimised_code" has combined ensemble_generator
and power_calculator. That directories version of the simulation is broken in so
far as the obtained AOUP plots do not reproduce the results within "Information
engine in a non-equilibrium bath" - Saha 2023.

@author: Lewis Dean - lewis.dean@manchester.student.ac.uk
"""

from ensemble_generator import generate_trajectory_ensemble
import model_constants as constants

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
    '''
    np.empty method that works well with Cython.
    '''
    return np.empty(shape, doubles)


cpdef double delta_free_energy(double lmb_n0, double lmb_n1, float delta_g):
    '''
    Returns the change in equilibrium free energy after an increment in
    trap center postion.
    ---------
    Returns:
        double
    '''
    return delta_g*(lmb_n1 - lmb_n0)


cpdef double calculate_trap_work(double x, double lmb_n0, double lmb_n1,
                                 float gain):
    '''
    Used to calculate the trap work associated with a ratchet event.
    If statement acts to implement Heaviside step function of feedback rule.
    ---------
    Returns:
        double
    '''
    return 0 if (lmb_n1 - lmb_n0) == 0 else 0.5*gain*(gain-2)*(x - lmb_n0)**2
    

cpdef cnp.ndarray[cnp.double_t, ndim=2] generate_power_ensembles(
    cnp.ndarray[cnp.double_t, ndim=3] trajectory_ensemble,
    float delta_g,
    float total_time,
    float gain,
    int samples=constants.SAMPLE_PATHS,
    int steps=constants.N,
    float transient_fraction=constants.TRANSIENT_FRACTION):
    '''
    This function takes the inputted trajectory ensemble and returns
    the associated power ensemble (averaged over engine trajectory).
    ---------
    Returns:
        2D Cython Numpy array of doubles
    '''
    cdef int sample_index, count
    cdef double x
    cdef int transient_steps = round(steps*transient_fraction)
    cdef float sampling_time = total_time*(1-transient_fraction)
    cdef cnp.ndarray[cnp.double_t, ndim=1] x_array, lmb_array, free_power_ensemble, trap_power_ensemble
    
    cdef cnp.ndarray[cnp.double_t, ndim=2] delta_free_energy_ensemble = (
        return_empty((samples, steps-transient_steps + 1)))
    cdef cnp.ndarray[cnp.double_t, ndim=2] trap_work_ensemble = (
        return_empty((samples, steps-transient_steps + 1)))
    
    
    for sample_index, trajectory in enumerate(trajectory_ensemble):
        x_array, lmb_array = trajectory[0], trajectory[1]
        
        for count, x in enumerate(x_array):
            if count < (steps - transient_steps):  
        
                delta_free_energy_ensemble[sample_index, count] = (
                    delta_free_energy(lmb_array[count], lmb_array[count+1],
                                      delta_g))
                trap_work_ensemble[sample_index, count] = (
                    calculate_trap_work(x, lmb_array[count],
                                        lmb_array[count+1], gain))
    
    free_power_ensemble = 1/(sampling_time) * np.sum(
        delta_free_energy_ensemble, axis=1)
    trap_power_ensemble = 1/(sampling_time) * np.sum(
        trap_work_ensemble, axis=1)
        
    cdef cnp.ndarray[cnp.double_t, ndim=2] power_results = (
        return_empty((2, samples)))
    power_results[0], power_results[1] = free_power_ensemble, trap_power_ensemble
        
    return power_results


cpdef plot_free_power(cnp.ndarray[cnp.double_t, ndim=1] free_powers,
                      float tau_c, float d_ne, float delta_g,
                      float total_time=constants.PROTOCOL_TIME,
                      float transient_fraction=constants.TRANSIENT_FRACTION,
                      int model=constants.NOISE_MODEL):
    '''
    This function produces a histogram from the ensemble of engine free power
    outputs.
    ---------
    Returns:
        None
    '''
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
    float tau_c, float d_ne, float gain, float delta_g,
    float time=constants.PROTOCOL_TIME):
    
    cdef cnp.ndarray[cnp.double_t, ndim=3] trajectory_ensemble = (
    generate_trajectory_ensemble(tau_c, d_ne, gain, delta_g))
    
    cdef cnp.ndarray[cnp.double_t, ndim=2] power_ensembles = (
    generate_power_ensembles(trajectory_ensemble, delta_g, time, gain))
    cdef cnp.ndarray[cnp.double_t, ndim=1] free_power_ensemble = (
        power_ensembles[0])
    cdef cnp.ndarray[cnp.double_t, ndim=1] trap_power_ensemble = (
        power_ensembles[1])

    cdef cnp.ndarray[cnp.double_t, ndim=1] mean_power_results = (
        return_empty(4))
    mean_power_results[0] = np.mean(free_power_ensemble)
    mean_power_results[1] = standard_error_on_mean(free_power_ensemble)
    mean_power_results[2] = np.mean(trap_power_ensemble)
    mean_power_results[3] = standard_error_on_mean(trap_power_ensemble)
            
    return mean_power_results


cpdef gaussian_test(float tau_c, float d_ne, float gain, float delta_g,
                    float time=constants.PROTOCOL_TIME):
    '''
    This was initially used to demonstrate the non-Gaussian nature of the 
    free power distribution - hence the need for bootstrapping to calculate
    a standard error on the mean, as opposed to taking a central limit theorem
    approach.
    ---------
    Returns:
        2D Cython Numpy array of doubles
    '''
    
    cdef cnp.ndarray[cnp.double_t, ndim=2] trajectory_ensemble = (
    generate_trajectory_ensemble(tau_c, d_ne, gain, delta_g))
    
    cdef cnp.ndarray[cnp.double_t, ndim=1] power_ensembles = (
    generate_power_ensembles(trajectory_ensemble, delta_g, time, gain))
    cdef cnp.ndarray[cnp.double_t, ndim=1] free_power_ensemble = (
        power_ensembles[0])
    
    plot_free_power(free_power_ensemble, tau_c, d_ne, delta_g)
    
    