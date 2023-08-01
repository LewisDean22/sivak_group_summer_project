# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:52:20 2023

low tau limit = white noise

@author: lewis
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
from free_power_calculator_cython import calculate_mean_free_power


IMAGE_DIRECTORY = "plots/"
DATA_DIRECTORY = "data_files/"

PLOT_FREQUENCY = 1
PLOT_CORRELATION_TIME = 0
PLOT_NOISE_STRENGTH = 0
SAVE_PLOT = 1

TRAJECTORIES_AVERAGED = 1000

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


def error_on_mean(std, n):
    
    return std / np.sqrt(n)


def fitting_model(x, a, b, c, d):
    '''
    The noise strength plots are likely some sort of power law.
    This is clear aactually by the model in Jannik's paper. Seeing
    that equation, it seems finding a fit may not be so trivial...
    
    The correlation time plots can certainly not be modelled by
    a simple polynomial. Could be a tanh, logistic or Gompertz curve.
    Maybe investigate all and see which fits best. Also arctan and erf functions.
    
    Now these functional forms need constants for scaling and translation.
    Be careful not to overfit however. Check the correlation between fit
    parameters (as seen on OriginPro). For example, could add a translational
    dof to Gompertz curve but will that overfit?

    '''
    
    # return a*np.tanh(x - b) + c
    # return a*np.exp(-b*np.exp(-c*x)) + d
    # return (a / (1 + np.exp(-b*(x-c)))) + d
    return a*erf(b*x - c) + d


def simulation_data_fitter(x, y, unc_y, parameter_estimates):

    popt, pcov = curve_fit(fitting_model, x, y,
                       p0 = parameter_estimates, sigma = unc_y)
    
    return popt, pcov


def plot_power_against_cutoff_frequency(f_ne, mean_free_power,
                                        unc_mean_free_power,
                                        low_f_ne_power, 
                                        fit_logspace,
                                        fitted_powers,
                                        d_ne, delta_g):
    
    fig, ax_1 = plt.subplots()
    label = ("Noise model 1: " + "\n" + 
             r"$D_{ne}$ = " + "{:.3g}".format(d_ne) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(delta_g))
    
    plt.title(r"$\langle \dot{F} \rangle$ as a function of $f_{ne}$",
              fontsize=18)
    
    ax_1.set_xscale("log")
    ax_1.set_xlabel(r"$f_{ne} \quad [1 \, / \, \tau_r]$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{F} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=25, fontsize=18)
    
    ax_1.errorbar(f_ne, mean_free_power, unc_mean_free_power, c='r', zorder=1,
                  fmt="None")
    ax_1.scatter(f_ne, mean_free_power, c='#0033cc', zorder=2,
                 label=label, s=5)
   
    # ax_1.plot(fit_logspace, fitted_powers, c='#42b0f5',
    #           label="Erf fit")
    ax_1.axhline(white_noise_limit_power(d_ne, delta_g),
                 c="#7F7F7F", linestyle="--")

    ax_1.axhline(low_f_ne_power,
                 c="#7F7F7F", linestyle="dotted", zorder=0)
    
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOT:
        
        plt.savefig(IMAGE_DIRECTORY + "mean_free_power_against_cutoff_frequency",
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
    ax_1.set_xlabel(r"$\tau_c \quad [\tau_r]$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{F} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=25, fontsize=18)

    
    ax_1.scatter(tau_c, mean_free_power, c='r', zorder=1,
                 label=label)

   
    ax_1.axhline(white_noise_limit_power(d_ne, delta_g),
                 c="#7F7F7F", linestyle="--")
    ax_1.axhline(f_ne_low_power_limit,
                 c="#7F7F7F", linestyle="dotted", zorder=0)
    
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
    label = ("Noise model 1: " + "\n" + 
             r"$f_{ne}$ = " + "{:.3g}".format(f_ne) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(delta_g))
    plt.title(r"$\langle \dot{F} \rangle$ as a function of $D_{ne}$",
              fontsize=18)
    
    max_power_tick = len(str(round(np.max(mean_free_power))))
    
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")
    ax_1.set_ylim(0.1, 10**max_power_tick)
    
    ax_1.set_xlabel(r"$D_{ne}$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{F} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=20, fontsize=18)

    ax_1.scatter(d_ne, mean_free_power, c='r', zorder=1, 
                 label=label)

    fit_logspace = np.logspace(np.log10(d_ne[0]), np.log10(d_ne[-1]),
                               num=100, base=10)
    ax_1.plot(fit_logspace, white_noise_limit_power(fit_logspace, delta_g),
              c='#42b0f5', label='Analytic fit')
    
    ax_1.axhline(low_f_ne_power,
                 c="#7F7F7F", linestyle="dotted", zorder=0)
    
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOT:
        
        plt.savefig(IMAGE_DIRECTORY + "server_mean_free_power_against_noise_strength",
                    dpi=600)
    
    plt.show()
    plt.close()
    

def main():
    
    if PLOT_FREQUENCY or PLOT_CORRELATION_TIME:
        
        frequency_data_analysis = np.genfromtxt(DATA_DIRECTORY +
                                                "power_against_cutoff_frequency_error_bars_model_1.csv",
                                                delimiter=',', skip_header=1)
        
        f_ne_logspace = frequency_data_analysis[:,0]
        tau_c_logspace = 1 / f_ne_logspace
        f_ne_mean_free_powers = frequency_data_analysis[:,1] 
        unc_free_power = frequency_data_analysis[:,2]
        unc_mean_free_power = error_on_mean(unc_free_power, TRAJECTORIES_AVERAGED)
        #f_ne_low_power_limit = f_ne_non_active_bath_power(D_NE, F_NE_DELTA_G)
        f_ne_low_power_limit = LOW_F_NE
    
        if PLOT_FREQUENCY:
            
            parameter_estimates = (0.2, 10000, 5, 0.3)
            popt, pcov = simulation_data_fitter(f_ne_logspace,
                                                f_ne_mean_free_powers,
                                                unc_mean_free_power,
                                                parameter_estimates)
            
            fit_logspace = np.logspace(np.log10(f_ne_logspace[0]),
                                       np.log10(f_ne_logspace[-1]), 100)
            fitted_powers = fitting_model(fit_logspace, *popt)
            
            plot_power_against_cutoff_frequency(f_ne_logspace,
                                                f_ne_mean_free_powers,
                                                unc_mean_free_power,
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
                                                      "power_against_noise_strength_model_1.csv",
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
