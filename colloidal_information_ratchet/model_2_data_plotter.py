# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:52:20 2023

Maybe the analytic curve should have a large logspace
to make it smoother

@author: lewis
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from free_power_calculator_cython import calculate_mean_free_power

IMAGE_DIRECTORY = "alternate_noise_plots/"
DATA_DIRECTORY = "alternate_noise_data_files/"

PLOT_FREQUENCY = 1
PLOT_CORRELATION_TIME = 0
PLOT_NOISE_STRENGTH = 0
SAVE_PLOT = 1

LOWER_TRUNC_FOR_FIT = 0
UPPER_TRUNC_FOR_FIT = -1

D_NE = 3
F_NE_DELTA_G = 0.37
LOW_F_NE = 0.2385624848325371
LOW_D_NE = 0.23418417007143202

F_NE = 118
D_NE_DELTA_G = 0.38


def tilde_delta_g(d_ne, delta_g):
    
    return delta_g / np.sqrt(1+d_ne)


def white_noise_limit_power(d_ne, delta_g):
    
    output_list = []
    
    tdg = tilde_delta_g(d_ne, delta_g)
    part_1 = (1+d_ne)*np.sqrt(2/np.pi)
    part_2 = tdg*np.exp(-tdg**2 / 2)
    
    if isinstance(tdg, np.ndarray):
        for count, val in enumerate(tdg):
            part_3 = 1 / (1 + math.erf(val/np.sqrt(2)))
            output_list += [part_1[count]*part_2[count]*part_3]
        return output_list
    
    part_3 = 1 / (1 + math.erf(tdg/np.sqrt(2)))
    return part_1*part_2*part_3


def f_ne_non_active_bath_power(d_ne, delta_g):
    
    return calculate_mean_free_power(0, d_ne, delta_g)


def d_ne_non_active_bath_power(f_ne, delta_g):
    
    return calculate_mean_free_power(0, f_ne, delta_g)


def fitting_model(x, a, b):
    '''    
    Now these functional forms need constants for scaling and translation.
    Be careful not to overfit however. Check the correlation between fit
    parameters (as seen on OriginPro).
    '''
    
    return (0.1 / np.cosh(a*x - b)**2) + 0.234


def simulation_data_fitter(x, y, unc_y, parameter_estimates):

    popt, pcov = curve_fit(fitting_model, x, y,
                       p0 = parameter_estimates, sigma = unc_y,
                       method='lm')
    
    return popt, pcov


def plot_power_against_cutoff_frequency(f_ne, mean_free_power, 
                                        low_f_ne_power,
                                        fit_logspace,
                                        fitted_powers,
                                        d_ne, delta_g):
    
    fig, ax_1 = plt.subplots()
    label = ("Noise model 2: " + "\n" + 
             r"$D_{ne}$ = " + "{:.3g}".format(d_ne) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(delta_g))
    
    plt.title(r"$\langle \dot{F} \rangle$ as a function of $f_{ne}$",
              fontsize=18)
    
    ax_1.set_xscale("log")
    # ax_2 = ax_1.twinx()
    ax_1.set_xlabel(r"$f_{ne} \quad [1 \, / \, \tau_r]$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{F} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=25, fontsize=18)
    # ax_2.set_ylabel(r"$\frac{\langle \dot{F} \rangle}{\langle \dot{F_0} \rangle}$",
    #                 rotation=0, labelpad=25, fontsize=18)
    
    # ax_1.axvline(f_ne[LOWER_TRUNC_FOR_FIT], c='k')
    # ax_1.axvline(f_ne[UPPER_TRUNC_FOR_FIT], c='k')
    
    ax_1.plot(fit_logspace, fitted_powers, c='#42b0f5',
              label=r"sech$^2$ fit")
    ax_1.scatter(f_ne, mean_free_power, c='r', zorder=1,
                 label=label)
    # ax_2.scatter(f_ne, mean_free_power/low_f_ne_power, c='r', zorder=1,)   
    
    # ax_2.plot(f_ne, mean_free_power/low_f_ne_power,
    #           c='#42b0f5', linestyle='-',
    #         zorder=0, label=label)
   

    
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOT:
        
        plt.savefig(IMAGE_DIRECTORY + "fitted_mean_free_power_against_cutoff_frequency_2",
                    dpi=600)
    
    plt.show()
    plt.close()
    

def plot_power_against_correlation_time(tau_c, mean_free_power, 
                                        f_ne_low_power_limit,
                                        d_ne, delta_g):
    
    fig, ax_1 = plt.subplots()
    label = ("Noise model 1: " + "\n" + 
             r"$D_{ne}$ = " + "{:.3g}".format(d_ne) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(delta_g))
    
    plt.title(r"$\langle \dot{F} \rangle$ as a function of $\tau_c$",
              fontsize=18)
    
    ax_1.set_xscale("log")
    # ax_2 = ax_1.twinx()
    ax_1.set_xlabel(r"$\tau_c \quad [\tau_r]$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{F} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=25, fontsize=18)
    # ax_2.set_ylabel(r"$\frac{\langle \dot{F} \rangle}{\langle \dot{F_0} \rangle}$",
    #                 rotation=0, labelpad=25, fontsize=18)
    
    ax_1.scatter(tau_c, mean_free_power, c='r', zorder=1,
                 label=label)
    # ax_2.scatter(f_ne, mean_free_power/low_f_ne_power, c='r', zorder=1,)   
    
    # ax_1.plot(f_ne, mean_free_power,
    #           c='#42b0f5', linestyle='-',
    #         zorder=0, label=label)
    # ax_2.plot(f_ne, mean_free_power/low_f_ne_power,
    #           c='#42b0f5', linestyle='-',
    #         zorder=0, label=label)
   
    ax_1.axhline(white_noise_limit_power(d_ne, delta_g),
                 c="#7F7F7F", linestyle="--")
    # ax_2.axhline(white_noise_limit_power(d_ne)/low_f_ne_power,
    #              c="#7F7F7F", linestyle="--")
    ax_1.axhline(f_ne_low_power_limit,
                 c="#7F7F7F", linestyle="dotted", zorder=0)
    # ax_2.axhline(1,
    #              c="#7F7F7F", linestyle="dotted")
    
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOT:
        
        plt.savefig(IMAGE_DIRECTORY + "mean_free_power_against_correlation_time",
                    dpi=600)
    
    plt.show()
    plt.close()
    

def plot_power_against_noise_strength(d_ne, mean_free_power,
                                      low_f_ne_power,
                                      f_ne, delta_g):
    
    fig, ax_1 = plt.subplots()
    label = ("Noise model 2: " + "\n" +
             r"$f_{ne}$ = " + "{:.3g}".format(f_ne) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(delta_g))
    plt.title(r"$\langle \dot{F} \rangle$ as a function of $D_{ne}$",
              fontsize=18)
    
    max_power_tick = len(str(round(np.max(mean_free_power))))*10
    
    # ax_2 = ax_1.twinx()
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")
    ax_1.set_ylim(0.1, max_power_tick)
    # ax_2.set_yscale("log")
    
    ax_1.set_xlabel(r"$D_{ne}$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{F} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=20, fontsize=18)
    # ax_2.set_ylabel(r"$\frac{\langle \dot{F} \rangle}{\langle \dot{F_0} \rangle}$",
    #                 rotation=0, labelpad=20, fontsize=18)

    ax_1.scatter(d_ne, mean_free_power, c='r', zorder=1, 
                 label=label)
    # ax_2.scatter(d_ne, mean_free_power/low_f_ne_power, c='r', zorder=1) 
    # ax_1.plot(d_ne, mean_free_power, c='#42b0f5', linestyle='-',
    #         zorder=0, label=label)
    # ax_1.plot(d_ne, white_noise_limit_power(d_ne, delta_g),
    #           c='#42b0f5', label='Analytic fit')
    # ax_2.plot(d_ne, mean_free_power/low_f_ne_power, c='#42b0f5',
    #           linestyle='-', zorder=0, label=label)
    
    ax_1.axhline(low_f_ne_power,
                 c="#7F7F7F", linestyle="dotted", zorder=0)
    
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOT:
        
        plt.savefig(IMAGE_DIRECTORY + "mean_free_power_against_noise_strength_2",
                    dpi=600)
    
    plt.show()
    plt.close()
    

def main():
    
    if PLOT_FREQUENCY or PLOT_CORRELATION_TIME:
        
        frequency_data_analysis = np.genfromtxt(DATA_DIRECTORY +
                                                "power_against_cutoff_frequency_model_2.csv",
                                                delimiter=',', skip_header=1)
        f_ne_logspace = frequency_data_analysis[:,0]
        tau_c_logspace = 1 / f_ne_logspace
        f_ne_mean_free_powers = frequency_data_analysis[:,1] 
        
        # # f_ne_low_power_limit = f_ne_non_active_bath_power(D_NE, F_NE_DELTA_G)
        f_ne_low_power_limit = LOW_F_NE
        
        if PLOT_FREQUENCY:
            
            parameter_estimates = (0.3, 1)
            popt, pcov = simulation_data_fitter(f_ne_logspace[LOWER_TRUNC_FOR_FIT:UPPER_TRUNC_FOR_FIT],
                                                f_ne_mean_free_powers[LOWER_TRUNC_FOR_FIT:UPPER_TRUNC_FOR_FIT],
                                                [0.01]*len(f_ne_logspace[LOWER_TRUNC_FOR_FIT:UPPER_TRUNC_FOR_FIT]),
                                                parameter_estimates)
            print(popt)
            
            fit_logspace = np.logspace(np.log10(f_ne_logspace[0]),
                                       np.log10(f_ne_logspace[-1]), 100)
            fitted_powers = fitting_model(fit_logspace, 2, 1)
            
            plot_power_against_cutoff_frequency(f_ne_logspace,
                                                f_ne_mean_free_powers,
                                                f_ne_low_power_limit,
                                                fit_logspace,
                                                fitted_powers,
                                                D_NE, F_NE_DELTA_G)
        
        if PLOT_CORRELATION_TIME:
           
            plot_power_against_correlation_time(tau_c_logspace,
                                                f_ne_mean_free_powers,
                                                f_ne_low_power_limit,
                                                D_NE, F_NE_DELTA_G)
        
    
    if PLOT_NOISE_STRENGTH:
        
        noise_strength_data_analysis = np.genfromtxt(DATA_DIRECTORY +
                                                      "power_against_noise_strength_model_2.csv",
                                                      delimiter=',', skip_header=1)
        d_ne_logspace = noise_strength_data_analysis[:,0]
        d_ne_mean_free_powers = noise_strength_data_analysis[:,1] 
        
        
        # d_ne_low_power_limit = d_ne_non_active_bath_power(F_NE, D_NE_DELTA_G)
        d_ne_low_power_limit = LOW_D_NE
        plot_power_against_noise_strength(d_ne_logspace,
                                          d_ne_mean_free_powers,
                                          d_ne_low_power_limit,
                                          F_NE, D_NE_DELTA_G)
        
  
if __name__ == '__main__':
    main()
