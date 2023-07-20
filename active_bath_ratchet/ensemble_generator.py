# -*- coding: utf-8 -*-
"""
Created on Tue Jul 4 18:57:42 2023

Is this a tightly coupled bipartite system between some
controller/memory upstream and the downstream system which
is confined to the trap. I say this as it seems that whenever
work is inputted, the system is bound to move forward. Although,
does it really or on average has it now moved forward.

Milestein method not needed because for addiive noise,
the Euler-Maruyama and Milstein methods have equivalent
strong orders of convergence.

Does this process involve a stiff SDE? Is there disparate
time scales for relaxation to thermal equilibrium and
the time between feedback updates?

Only free energy is a useful power output here as the negative
trap work is dissipated as heat to the colloidal medium. Hence,
the zero work feedback scheme will be used here, with alpha=2.

Is the colloidal medium a perfect thermal resevoir or, if heat is
continually dissipated by a feedback protocol, would the
temp increase and thus the ability of the trap potential to
confine the colloid decrease.

@author: lewis
"""

import numpy as np
from joblib import Parallel, delayed
from scipy.constants import g, k #noqa
import active_bath_constants as constants


NUMBER_OF_CORES_USED = -1


def drift_term(x, lmb, delta_g):
    
    return -(x-lmb) - delta_g


def diffusion_term(dt):
    
    return np.sqrt(2*dt)


def noise_drift_term(zeta, f_ne):
    
    return -zeta*f_ne
    

def noise_diffusion_term(f_ne, d_ne, dt):
    
    return f_ne*np.sqrt(2*d_ne*dt)


def wiener_increment():
    
    return np.random.normal(0, 1)


def delta_z(zeta, f_ne, d_ne, dt, dw):
    
    return noise_drift_term(zeta, f_ne)*dt + noise_diffusion_term(
        f_ne, d_ne, dt)*dw


def delta_x(x, lmb, delta_g, zeta, dt, dw):
    
    return (drift_term(x, lmb, delta_g) + zeta)*dt + diffusion_term(dt) * dw


def heun_method(f_ne, d_ne, delta_g,
                x_init=constants.X_INIT,
                dt=constants.DELTA_T,
                lmb_init=constants.LAMBDA_INIT,
                steps=constants.N,
                freq_measure=constants.MEASURING_FREQUENCY,
                threshold=constants.THRESHOLD,
                gain=constants.GAIN,
                offset=constants.OFFSET):

    
    time = 0
    x = x_init
    zeta = 0
    lmb_array = [lmb_init]
    
    for n in range(steps):
        lmb = lmb_array[-1]
        
        dw_zeta = wiener_increment()
        zeta_estimate = zeta + delta_z(zeta, f_ne, d_ne, dt, dw_zeta)
        zeta = (zeta + 0.5*dt*(noise_drift_term(zeta, f_ne) + 
                            noise_drift_term(zeta_estimate, f_ne)) + 
                noise_diffusion_term(f_ne, d_ne, dt)*dw_zeta)
                
           
        dw_x = wiener_increment()
        delta_x_n = delta_x(x, lmb, delta_g, zeta, dt, dw_x)
        x = x + delta_x_n
        
        time = (n+1)*dt
        
        if ((time*freq_measure).is_integer() and 
            (x - lmb) > threshold):
            lmb_array += [lmb + gain*(x - lmb) + offset]
        else:
            lmb_array += [lmb]
        
    
    return lmb_array


def sample_path_generator(i, f_ne, d_ne, delta_g):
        
        return  heun_method(f_ne, d_ne, delta_g)


def generate_trajectory_ensemble(f_ne, d_ne, delta_g,
                                 samples=constants.SAMPLE_PATHS,
                                 n_cores=NUMBER_OF_CORES_USED):
    
    well_positions = Parallel(n_jobs=n_cores,
                              backend = "threading",
                              verbose=0)(
    delayed(sample_path_generator)(i, f_ne, d_ne, delta_g) for i in range(samples))

    return well_positions
