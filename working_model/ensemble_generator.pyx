# -*- coding: utf-8 -*-
# cython: language_level=3
"""
This script takes the defined model constants and applys them to
the model of the colloidal information engine to produce an ensemble
of engine trajectories. Each trajectory contains all simulated x and lambda
values, however doing so demands much more memory than is necessary.

The approach taken in the directory "broken_but_more_optimised_code"
implements a simpler, and hence more memory-efficient, way of calculating
the mean free power output for a given set of active noise parameters.

@author: Lewis Dean - lewis.dean@manchester.student.ac.uk
"""

import numpy as np
cimport numpy as cnp
import model_constants as constants

cnp.import_array()
doubles = np.float64
ctypedef cnp.double_t doubles_t


cpdef return_empty(shape):
    
    return np.empty(shape, doubles)


cpdef double drift_term(double x, double lmb, float delta_g):
    '''
    Returns drift term in SDE for x evolution.
    ---------
    Returns:
        double
    '''
    return -(x-lmb) - delta_g


cpdef double equilibrium_diffusion_term(double dt=constants.DELTA_T):
    '''
    Returns thermal noise diffusion term in SDE for x evolution.
    ---------
    Returns:
        double
    '''
    return np.sqrt(2*dt)


cpdef double noise_drift_term(double zeta, float tau_c):
    '''
    Returns drift term in SDE for active noise (zeta) evolution.
    ---------
    Returns:
        double
    '''
    return -zeta / tau_c


cpdef double noise_diffusion_term(float tau_c, float d_ne,
                                  double dt=constants.DELTA_T,
                                  int model=constants.NOISE_MODEL):
    '''
    Returns diffusion term in SDE for active noise (zeta) evolution.
    ---------
    Returns:
        double
    '''
    if model == 1:
        return np.sqrt(2*d_ne*dt) / tau_c
    elif model == 2:
        return np.sqrt(2*d_ne*dt/tau_c)


cpdef float wiener_increment():
    '''
    Calculates a random variable, proportional to a Wiener increment,
    by drawing from the standard normal distribtion. Delta_t variance
    of the Wiener increment factorised out into the associated diffusion
    term's prefactor.
    ---------
    Returns:
        float
    '''
    return np.random.normal(0, 1)


cpdef float generate_initial_active_noise(
    float tau_c, float d_ne,
    int model=constants.NOISE_MODEL,
    int equilibrium_noise=constants.EQUILIBRIUM_NOISE_INITIALISATION):
    '''
    This function initialises the active noise by drawing from its
    stationary distribution (shown in project report appendices).
    ---------
    Returns:
        float
    '''
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
    '''
    This function initialises x by drawing from its
    stationary distribution.
    ---------
    Returns:
        float
    '''
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
    '''
    Calculates the next increment in active noise via the Euler-Maruyama
    method.
    ---------
    Returns:
        double
    '''
    return noise_drift_term(zeta, tau_c)*dt + noise_diffusion_term(
        tau_c, d_ne)*dw
    

cpdef double zeta_heun_method(double zeta, float tau_c, float d_ne,
                              double dt=constants.DELTA_T,
                              int model=constants.NOISE_MODEL):
    '''
    This function implements the Heun's method, using the delta_zeta
    function to provide an estimator, to find a subsequent zeta with 
    stronger convergence.
    ---------
    Returns:
        double
    '''
    cdef double dw_zeta, zeta_estimate
 
    dw_zeta = wiener_increment()
    zeta_estimate = zeta + delta_zeta(zeta, tau_c, d_ne, dw_zeta)
         
    zeta = (zeta + 0.5*dt*(noise_drift_term(zeta, tau_c) + 
            noise_drift_term(zeta_estimate, tau_c)) + 
            noise_diffusion_term(tau_c, d_ne)*dw_zeta)
         
    return zeta
     

cpdef double delta_x(double x, double lmb, float delta_g,
                     double zeta, double dw_x, double dt=constants.DELTA_T):
    '''
    Calculates the next increment in x via the Euler-Maruyama
    method.
    ---------
    Returns:
        double
    '''
    return (drift_term(x, lmb, delta_g) + zeta)*dt + equilibrium_diffusion_term() * dw_x


cpdef cnp.ndarray[cnp.double_t, ndim=2] x_euler_method(
    float tau_c, float d_ne, float gain,
    float delta_g, int steps=constants.N,
    double freq_measure=constants.MEASURING_FREQUENCY,
    double dt=constants.DELTA_T,
    float threshold=constants.THRESHOLD,
    float offset=constants.OFFSET,
    float transient_fraction=constants.TRANSIENT_FRACTION):
    '''
    This function combines the numerical methods defined above to calculate
    the evolution of both zeta and x over the specified simulation time.
    ---------
    Returns:
        2D Cython Numpy array of doubles
    '''
    cdef int n, i
    cdef bint transience_finished
    cdef double time, zeta, x, lmb, dw_x
    cdef int transient_steps = round(steps*transient_fraction)
    cdef cnp.ndarray[cnp.double_t, ndim=1] x_array = (
        return_empty(steps - transient_steps +1))
    cdef cnp.ndarray[cnp.double_t, ndim=1] lmb_array = (
        return_empty(steps-transient_steps+1))
    
    transience_finished = 0
    time = 0
    x = generate_initial_x(tau_c, d_ne, delta_g)
    lmb = 0
    zeta = generate_initial_active_noise(tau_c, d_ne)
    
    for n in range(steps):
        
        zeta = zeta_heun_method(zeta, tau_c, d_ne)
        
        if n < transient_steps:
                
            dw_x = wiener_increment()
            x = x + delta_x(x, lmb, delta_g, zeta, dw_x)
            
            time = (n+1)*dt
            
            if ((time*freq_measure).is_integer() and 
                (x - lmb) > threshold):
                lmb = lmb + gain*(x - lmb) + offset
            
        else:
            
            if not transience_finished:
                x_array[0] = x
                lmb_array[0] = lmb
                transience_finished = 1
    
            i = n - transient_steps
            x, lmb = x_array[i], lmb_array[i]
                
            dw_x = wiener_increment()
            x_array[i+1] = x + delta_x(x, lmb, delta_g, zeta, dw_x)
            
            time = (n+1)*dt
            
            if ((time*freq_measure).is_integer() and 
                (x - lmb) > threshold):
                lmb_array[i+1] = lmb + gain*(x - lmb) + offset
            else:
                lmb_array[i+1] = lmb
                
    return np.array([x_array, lmb_array])


cpdef cnp.ndarray[cnp.double_t, ndim=3] generate_trajectory_ensemble(
    float tau_c, float d_ne, float gain,
    float delta_g, int steps=constants.N,
    int samples=constants.SAMPLE_PATHS,
    float transient_fraction=constants.TRANSIENT_FRACTION):
    '''
    Main function of the script which finally produces the ensemble of
    engine trajectories
    ---------
    Returns:
        3D Cython Numpy array of doubles
    '''
    cdef int j
    cdef int transient_steps = round(steps*transient_fraction)
    cdef cnp.ndarray[cnp.double_t, ndim=3] trajectories = (
        return_empty((samples, 2, steps-transient_steps+1)))
    
    for j in range(samples):
        trajectories[j] = x_euler_method(tau_c, d_ne, gain, delta_g)
    
    return trajectories
