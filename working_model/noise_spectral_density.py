# -*- coding: utf-8 -*-
"""

Transient period to allow noise to equilbriate? How can I justify this?

F_NE of 10**4 is the white noise limit. F_NE of 10**-4 is
very correlated noise. Opposite for TAU_C.

Correlation time and f_ne reciprocal quantites. Fourier transform
converts from one space to the other?

Joblib seems to work with cython in this instance...

Need to properly understand what the fast fourier transform
function does.

Also linearise the autocorrelation plot to prove exponetial nature -
use errors to fit a straight line to the curve

@author: Lewis Dean
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import model_constants as constants
from ensemble_generator import zeta_heun_method


IMPORT_FILE = "autocorrelation_data_tau_c_0.001_model_1.csv"

IMAGE_DIRECTORY = "plots/"
DATA_DIRECTORY = "data_files/"
GENERATE_DATA = 1
SAVE_DATA = 1

# Be careful changing these for old data...
TAU_C = 10**-3
D_NE = 3
DELTA_G = 0.37

PLOT_AUTOCORRELATION = 1
PLOT_PSD = 1
SAVE_PLOTS = 1

NUMBER_OF_CORES_USED = -1


def identify_data_model(data_filename):
    
    parameters = re.findall(r'\d+(?:\.\d+)?', data_filename)
    model_type = parameters[-1]
    
    return int(model_type)


def calculate_burn_in_steps(steps=constants.N,
                            transient_fraction=constants.TRANSIENT_FRACTION):

    return round(steps*transient_fraction)


def generate_time_array(dt=constants.DELTA_T, steps=constants.N):
    
    return np.linspace(0, steps*dt, steps+1)


def calculate_noise_evolution(tau_c, d_ne, delta_g,
                              steps=constants.N):
    
    zeta_array = [0]   
    for n in range(steps):
        zeta_array += [zeta_heun_method(zeta_array[-1], tau_c,
                                        d_ne, delta_g)]
    
    return zeta_array


def noise_path_generator(thread, tau_c, d_ne, delta_g):
        
    return  calculate_noise_evolution(tau_c, d_ne, delta_g)
    
    
def generate_noise_ensemble(tau_c, d_ne, delta_g,
                            samples=constants.SAMPLE_PATHS,
                            n_cores=NUMBER_OF_CORES_USED):
    
    noise_trajectories = Parallel(n_jobs=n_cores,
                                  backend = "threading", verbose=1)(
                                  delayed(noise_path_generator)(
                                  thread, tau_c,
                                  d_ne, delta_g) for thread in range(samples))

    return np.array(noise_trajectories)


def calculate_time_delays(time_array, transient_period):
    
    return time_array[transient_period:] - time_array[transient_period]


def calculculate_autocorrelations(noise_trajectories, transient_period,
                                 samples=constants.SAMPLE_PATHS,
                                 steps=constants.N):
    
    truncated_noise_trajectories = noise_trajectories[:, transient_period:]
    autocorrelation_data_length = steps - transient_period + 1
    zeta_products = np.empty((samples, autocorrelation_data_length))
    
    for sample_index, zeta_array in enumerate(truncated_noise_trajectories):
        for zeta_index, zeta in enumerate(zeta_array):

                zeta_products[sample_index, zeta_index] = zeta * zeta_array[0]
    
    zeta_autocorrelations = np.mean(zeta_products, axis=0)
    
    return zeta_autocorrelations


def plot_autocorrelation(time_delays, autocorrelations, model_name,
                         tau_c=TAU_C, d_ne=D_NE, delta_g=DELTA_G):
    
    plt.title((r"$\langle \zeta(t)\zeta(t + \tau) \rangle$ as a " +
               "function of $\tau$"), fontsize=14)
    plt.xlabel(r"$|\tau|$", fontsize=14)
    plt.ylabel(r"$\langle \zeta(t)\zeta(t + \tau) \rangle$", fontsize=14)
    label = ("{}: ".format(model_name) + "\n" + 
             r"$D_{ne}$ = " + "{}".format(d_ne) + "\n" + 
             r"$\delta_g$ = " + "{}".format(delta_g))
    plt.plot(time_delays, autocorrelations, label=label)
    plt.axvline(tau_c, c='r', linestyle='dotted', alpha=0.8,
                label=r"$\tau_c$ = " + "{:.0e}".format(tau_c))
    plt.xlim(0, 10*tau_c)
    plt.legend()
    
    if SAVE_PLOTS:
        plt.savefig(IMAGE_DIRECTORY + 
                    "noise_autocorrelation_plot_model_{}".format(
                        model_name), dpi=600)
    
    plt.show()
    plt.close()
        

def plot_linearised_autocorrelation(time_delays, autocorrelations, model_name,
                                    tau_c=TAU_C, d_ne=D_NE, delta_g=DELTA_G):
    '''
    Errors emerge due to logarithm of zero - need greater number of samples
    to smooth out errors in the exponetial autocorrelation curve.
    '''
    
    plt.title((r"$\langle \zeta(t)\zeta(t + \tau) \rangle$ as a " +
               "function of $\tau$"), fontsize=14)
    plt.xlabel(r"$|\tau|$", fontsize=14)
    plt.ylabel(r"$\langle \zeta(t)\zeta(t + \tau) \rangle$", fontsize=14)
    label = ("{}: ".format(model_name) + "\n" + 
             r"$D_{ne}$ = " + "{}".format(d_ne) + "\n" + 
             r"$\delta_g$ = " + "{}".format(delta_g))
    plt.plot(time_delays, np.log(autocorrelations), label=label)
    plt.axvline(tau_c, c='r', linestyle='dotted', alpha=0.8,
                label=r"$\tau_c$ = " + "{:.0e}".format(tau_c))
    plt.xlim(0, 10*tau_c)
    plt.legend()
    
    if SAVE_PLOTS:
        plt.savefig(IMAGE_DIRECTORY + 
                    "linearised_noise_autocorrelation_plot_model_{}".format(
                        model_name), dpi=600)
    
    plt.show()
    plt.close()
        

def plot_power_spectral_density(freq, f_transform, model_name,
                                tau_c=TAU_C, d_ne=D_NE, delta_g=DELTA_G):
    
        plt.title("Nonequilibrium noise power spectral density")
        plt.xlim((-15, 15))
        plt.xlabel(r"$f$", fontsize=14)
        plt.ylabel("Power spectral density", fontsize=14)
        label = ("{}: ".format(model_name) + "\n" + 
                 r"$\tau_c$ = " + "{}".format(tau_c) + "\n" + 
                 r"$D_{ne}$ = " + "{}".format(d_ne) + "\n" + 
                 r"$\delta_g$ = " + "{}".format(delta_g))
        plt.plot(freq, f_transform, label=label)
        plt.legend()
        
        if SAVE_PLOTS:
            plt.savefig(IMAGE_DIRECTORY +
                        "noise_spectral_density_plot_model_{}".format(
                            model_name), dpi=600)
        
        plt.show()
        plt.close()


def main(tau_c=TAU_C, d_ne=D_NE, delta_g=DELTA_G,
         delta_t=constants.DELTA_T, model=constants.NOISE_MODEL):
    
    if GENERATE_DATA:
        
        time = generate_time_array()
        transient_steps = calculate_burn_in_steps()
        time_delays = calculate_time_delays(time, transient_steps)
        noise_trajectories = generate_noise_ensemble(tau_c, d_ne, delta_g)
        
        autocorrelations = calculculate_autocorrelations(
            noise_trajectories, transient_steps)
        
        if model == 1:
            model_name = "AOUP"
        elif model == 2:
            model_name = "PLOUP"
        
        if SAVE_DATA:
            
            file = np.zeros((len(autocorrelations), 2))
            file[:,0] = time_delays
            file[:,1] = autocorrelations
            headers = ["time_delay", "autocorrelation"]
            df = pd.DataFrame(file, columns=headers)
            df.to_csv((DATA_DIRECTORY +
                      "autocorrelation_data_tau_c_{0}_model_{1}.csv".format(
                          tau_c, model)), index=0, header=1, sep=',')
            
            
    else:
           
        data = np.genfromtxt(DATA_DIRECTORY + IMPORT_FILE, delimiter=',',
                             skip_header=1)
        time_delays = data[:,0]
        autocorrelations = data[:,1]
        
        model = identify_data_model(IMPORT_FILE)
        if model == 1:
            model_name = "AOUP"
        elif model == 2:
            model_name = "PLOUP"
        
    tau_c_autocorrelation = autocorrelations[np.argmin(
        abs(time_delays - tau_c))]
    print("e-folded autocorrelation is {:.5f}.".format(
        autocorrelations[0]/np.exp(1)))
    print("Autocorrelation at correlation time is {:.5f}.".format(
        tau_c_autocorrelation))

    if PLOT_AUTOCORRELATION:
        
        plot_autocorrelation(time_delays, autocorrelations, model_name)
        plot_linearised_autocorrelation(time_delays, autocorrelations,
                                        model_name)
    
    if PLOT_PSD:
        
        f_transform = np.fft.fft(autocorrelations)
        freq = np.fft.fftfreq(len(autocorrelations), delta_t)
        plot_power_spectral_density(freq, f_transform, model_name)
        
    return None


if __name__ == '__main__':
    main()
