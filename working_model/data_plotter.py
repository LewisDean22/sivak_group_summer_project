# -*- coding: utf-8 -*-
"""

low tau limit = white noise
Are error bars on vertical log scale working correctly?
Note: there should also be error bars for the low power limits

on legend for model 2 correlation time plots, use np.argmax
to find optimum correlation times

Switch plot format to SVG for the report

@author: Lewis Dean
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from power_calculator import calculate_mean_powers


IMAGE_DIRECTORY = "plots/"
DATA_DIRECTORY = "data_files/"
TAU_C_FILENAME = "power_against_correlation_time_gain_2.5_model_2.csv"
D_NE_FILENAME = "power_against_noise_strength_gain_2.5_model_2.csv"
GAIN_FILENAME = "power_against_feedback_gain_model_2.csv"

PLOT_CORRELATION_TIME_DATA = 0
PLOT_NOISE_STRENGTH_DATA = 0
PLOT_GAIN_DATA = 1
SAVE_PLOTS = 1

# Lower limit changes with choice of feedback gain!
CALCULATE_LOW_LIMITS = 1
LOW_TAU_C = 0.21853978662550902
LOW_D_NE = 0.22197723023697405

# Chosen parameters for inputted tau_c data
TAU_C_D_NE = 3
TAU_C_DELTA_G = 0.37

# Chosen parameters for inputted d_ne data
D_NE_TAU_C = 10**-4
D_NE_DELTA_G = 0.37

# Chosen parameters for inputted gain data
GAIN_TAU_C = 10**-4
GAIN_D_NE = 0
GAIN_DELTA_G = 0.37


def identify_data_parameters(data_filename):
    
    parameters = re.findall(r'\d+(?:\.\d+)?', data_filename)
    
    if len(parameters) > 1:
        feedback_gain = parameters[0]
        noise_model = parameters[1]
        
        return float(feedback_gain), int(noise_model)
    
    else:
        noise_model = parameters[0]
        
        return int(noise_model)


def tilde_delta_g(d_ne, delta_g):
    
    return delta_g / np.sqrt(1+d_ne)


def white_noise_limit_power(d_ne, delta_g):
    
    tdg = tilde_delta_g(d_ne, delta_g)
    part_1 = (1+d_ne)*np.sqrt(2/np.pi)
    part_2 = tdg*np.exp(-tdg**2 / 2)
    part_3 = 1 / (1 + erf(tdg/np.sqrt(2)))
    
    return part_1*part_2*part_3


def tau_c_non_active_bath_powers(d_ne, gain, delta_g):
    
    return calculate_mean_powers(0, 0, gain, delta_g)[0]
 

def d_ne_non_active_bath_powers(tau_c, gain, delta_g):
    
    return calculate_mean_powers(tau_c, 0, gain, delta_g)[0]
    

def plot_free_power_against_correlation_time(tau_c, mean_free_power, 
                                             unc_tau_c_mean_free_power,
                                             tau_c_low_power_limit,
                                             tau_c_gain, model_name,
                                             d_ne, delta_g):
    
    fig, ax_1 = plt.subplots()
    label = ("{}: ".format(model_name) + "\n" + 
             r"$D_{ne}$ = " + "{:.3g}".format(d_ne) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(delta_g) + '\n' +
             r"$\alpha$ = " + "{:.2g}".format(tau_c_gain))
    
    plt.title(r"$\langle \dot{F} \rangle$ as a function of $\tau_c$",
              fontsize=18)
    
    ax_1.set_xscale("log")
    ax_1.set_xlabel(r"$\tau_c \quad [\tau_r]$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{F} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=25, fontsize=18)

    ax_1.errorbar(tau_c, mean_free_power, unc_tau_c_mean_free_power,
                  c='r', zorder=1, fmt="None")
    ax_1.scatter(tau_c, mean_free_power, c='#0033cc', zorder=2,
                 label=label, s=5)

    ax_1.axhline(tau_c_low_power_limit,
                 c="#7F7F7F", linestyle="dotted", zorder=0)
    
    if tau_c_gain == 2:
        ax_1.axhline(white_noise_limit_power(d_ne, delta_g),
                     c="#7F7F7F", linestyle="--")
    
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOTS:
        
        plt.savefig(IMAGE_DIRECTORY + model_name +
                    "_mean_free_power_against_correlation_time",
                    dpi=600)
    
    plt.show()
    plt.close()
    

def plot_trap_power_against_correlation_time(tau_c, mean_trap_power, 
                                             unc_tau_c_mean_trap_power,
                                             tau_c_gain, model_name,
                                             d_ne, delta_g):
    
    fig, ax_1 = plt.subplots()
    label = ("{}: ".format(model_name) + "\n" + 
             r"$D_{ne}$ = " + "{:.3g}".format(d_ne) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(delta_g) + '\n' +
             r"$\alpha$ = " + "{:.2g}".format(tau_c_gain))
    
    plt.title(r"$\langle \dot{W}_{trap} \rangle$ as a function of $\tau_c$",
              fontsize=18)
    
    ax_1.set_xscale("log")
    ax_1.set_xlabel(r"$\tau_c \quad [\tau_r]$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{W}_{trap} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=25, fontsize=18)

    ax_1.errorbar(tau_c, mean_trap_power, unc_tau_c_mean_trap_power,
                  c='r', zorder=1, fmt="None")
    ax_1.scatter(tau_c, mean_trap_power, c='#0033cc', zorder=2,
                 label=label, s=5)

    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOTS:
        
        plt.savefig(IMAGE_DIRECTORY + model_name + 
                    "_mean_trap_power_against_correlation_time",
                    dpi=600)
    
    plt.show()
    plt.close()
    

def plot_net_power_against_correlation_time(tau_c, mean_net_power, 
                                            unc_tau_c_mean_net_power,
                                            tau_c_gain, model_name,
                                            d_ne, delta_g):
        
    fig, ax_1 = plt.subplots()
    label = ("{}: ".format(model_name) + "\n" + 
             r"$D_{ne}$ = " + "{:.3g}".format(d_ne) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(delta_g) + '\n' +
             r"$\alpha$ = " + "{:.2g}".format(tau_c_gain))
    
    plt.title(r"$\langle P_{net} \rangle$ as a function of $\tau_c$",
              fontsize=18)
    
    ax_1.set_xscale("log")
    ax_1.set_xlabel(r"$\tau_c \quad [\tau_r]$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle P_{net} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=25, fontsize=18)

    ax_1.errorbar(tau_c, mean_net_power, unc_tau_c_mean_net_power,
                  c='r', zorder=1, fmt="None")
    ax_1.scatter(tau_c, mean_net_power, c='#0033cc', zorder=2,
                 label=label, s=5)

    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOTS:
        
        plt.savefig(IMAGE_DIRECTORY + model_name +
                    "_mean_net_power_against_correlation_time",
                    dpi=600)
    
    plt.show()
    plt.close()
    

def plot_free_power_against_noise_strength(d_ne, mean_free_power,
                                           unc_d_ne_mean_free_power,
                                           tau_c_low_power_limit,
                                           d_ne_gain, model_name,
                                           tau_c, delta_g):
    
    fig, ax_1 = plt.subplots()
    label = ("{}: ".format(model_name) + "\n" + 
             r"$\tau_c$ = " + "{:.0e}".format(tau_c) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(delta_g) + '\n' +
             r"$\alpha$ = " + "{:.2g}".format(d_ne_gain))
    
    plt.title(r"$\langle \dot{F} \rangle$ as a function of $D_{ne}$",
              fontsize=18)
    
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")
    ax_1.set_xlabel(r"$D_{ne}$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{F} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=20, fontsize=18)

    ax_1.errorbar(d_ne, mean_free_power, unc_d_ne_mean_free_power,
                  c='r', zorder=1, fmt="None")
    ax_1.scatter(d_ne, mean_free_power, c='#0033cc', zorder=2,
                 label=label, s=5)
    
    if d_ne_gain == 2:
        fit_logspace = np.logspace(np.log10(d_ne[0]), np.log10(d_ne[-1]),
                                   num=100, base=10)
        ax_1.plot(fit_logspace, white_noise_limit_power(fit_logspace, delta_g),
                  c='#ff9900', label='Analytic fit', zorder=0)
    
    ax_1.axhline(tau_c_low_power_limit,
                 c="#7F7F7F", linestyle="dotted", zorder=0)
    
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOTS:
        
        plt.savefig(IMAGE_DIRECTORY + model_name +
                    "_log_mean_free_power_against_noise_strength",
                    dpi=600)
    
    plt.show()
    plt.close()
    
    
def plot_trap_power_against_noise_strength(d_ne, mean_trap_power,
                                           unc_d_ne_mean_trap_power,
                                           d_ne_gain, model_name,
                                           tau_c, delta_g):
    
    fig, ax_1 = plt.subplots()
    label = ("{}: ".format(model_name) + "\n" + 
             r"$\tau_c$ = " + "{:.0e}".format(tau_c) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(delta_g) + '\n' +
             r"$\alpha$ = " + "{:.2g}".format(d_ne_gain))
    
    plt.title(r"$\langle \dot{W}_{trap} \rangle$ as a function of $D_{ne}$",
              fontsize=18)
    
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")    
    ax_1.set_xlabel(r"$D_{ne}$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{W}_{trap} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=20, fontsize=18)

    ax_1.errorbar(d_ne, mean_trap_power, unc_d_ne_mean_trap_power,
                  c='r', zorder=1, fmt="None")
    ax_1.scatter(d_ne, mean_trap_power, c='#0033cc', zorder=2,
                 label=label, s=5)
    
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOTS:
        
        plt.savefig(IMAGE_DIRECTORY + model_name + 
                    "_log_mean_trap_power_against_noise_strength",
                    dpi=600)
    
    plt.show()
    plt.close()
    
    
def plot_net_power_against_noise_strength(d_ne, mean_net_power,
                                          unc_d_ne_mean_net_power,
                                          d_ne_gain, model_name,
                                          tau_c, delta_g):
    
    fig, ax_1 = plt.subplots()
    label = ("{}: ".format(model_name) + "\n" + 
             r"$\tau_c$ = " + "{:.0e}".format(tau_c) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(delta_g) + '\n' +
             r"$\alpha$ = " + "{:.2g}".format(d_ne_gain))
    
    plt.title(r"$\langle P_{net} \rangle$ as a function of $D_{ne}$",
              fontsize=18)
    
    ax_1.set_xscale("log")
    ax_1.set_yscale("log")
    ax_1.set_xlabel(r"$D_{ne}$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle P_{net} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=20, fontsize=18)

    ax_1.errorbar(d_ne, mean_net_power, unc_d_ne_mean_net_power,
                  c='r', zorder=1, fmt="None")
    ax_1.scatter(d_ne, mean_net_power, c='#0033cc', zorder=2,
                 label=label, s=5)
    
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOTS:
        
        plt.savefig(IMAGE_DIRECTORY + model_name + 
                    "_log_mean_net_power_against_noise_strength",
                    dpi=600)
    
    plt.show()
    plt.close()
    
    
def plot_free_power_against_gain(gain, mean_free_power,
                                 unc_gain_mean_free_power, model_name,
                                 GAIN_TAU_C, GAIN_D_NE, GAIN_DELTA_G):
    
    fig, ax_1 = plt.subplots()
    label = ("{}: ".format(model_name) + "\n" + 
             r"$\tau_c$ = " + "{:.0e}".format(GAIN_TAU_C) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(GAIN_DELTA_G) + '\n' +
             r"$D_{ne}$ = " + "{:.2g}".format(GAIN_D_NE))
    
    plt.title(r"$\langle \dot{F} \rangle$ as a function of $\alpha$",
              fontsize=18)
    
    ax_1.set_xscale("linear")
    ax_1.set_yscale("linear")
    ax_1.set_xlabel(r"$\alpha$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{F} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=20, fontsize=18)

    ax_1.errorbar(gain, mean_free_power, unc_gain_mean_free_power,
                  c='r', zorder=1, fmt="None")
    ax_1.scatter(gain, mean_free_power, c='#0033cc', zorder=2,
                 label=label, s=5)
    
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOTS:
        
        plt.savefig(IMAGE_DIRECTORY + model_name +
                    "_mean_free_power_against_gain",
                    dpi=600)
    
    plt.show()
    plt.close()
    
    
def plot_trap_power_against_gain(gain, mean_trap_power,
                                 unc_gain_mean_trap_power, model_name,
                                 GAIN_TAU_C, GAIN_D_NE, GAIN_DELTA_G):
    
    fig, ax_1 = plt.subplots()
    label = ("{}: ".format(model_name) + "\n" + 
             r"$\tau_c$ = " + "{:.0e}".format(GAIN_TAU_C) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(GAIN_DELTA_G) + '\n' +
             r"$D_{ne}$ = " + "{:.2g}".format(GAIN_D_NE))
    
    plt.title(r"$\langle \dot{W}_{trap} \rangle$ as a function of $\alpha$",
              fontsize=18)
    
    ax_1.set_xscale("linear")
    ax_1.set_yscale("linear")
    ax_1.set_xlabel(r"$\alpha$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{W}_{trap} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=20, fontsize=18)

    ax_1.errorbar(gain, mean_trap_power, unc_gain_mean_trap_power,
                  c='r', zorder=1, fmt="None")
    ax_1.scatter(gain, mean_trap_power, c='#0033cc', zorder=2,
                 label=label, s=5)
    ax_1.axhline(0, label="Zero trap work", linestyle="dotted",
                 c="#7F7F7F", zorder=0)
    
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOTS:
        
        plt.savefig(IMAGE_DIRECTORY + model_name +
                    "_mean_trap_power_against_gain",
                    dpi=600)
    
    plt.show()
    plt.close()
    
    
def plot_net_power_against_gain(gain, mean_net_power,
                                unc_gain_mean_net_power, model_name,
                                GAIN_TAU_C, GAIN_D_NE, GAIN_DELTA_G):
    
    fig, ax_1 = plt.subplots()
    label = ("{}: ".format(model_name) + "\n" + 
             r"$\tau_c$ = " + "{:.0e}".format(GAIN_TAU_C) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(GAIN_DELTA_G) + '\n' +
             r"$D_{ne}$ = " + "{:.2g}".format(GAIN_D_NE))
    
    plt.title(r"$\langle P_{net} \rangle$ as a function of $\alpha$",
              fontsize=18)
    
    ax_1.set_xscale("linear")
    ax_1.set_yscale("linear")
    ax_1.set_xlabel(r"$\alpha$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle P_{net} \rangle \quad [K_B T \, / \, \tau_r]$",
                    rotation=90, labelpad=20, fontsize=18)

    ax_1.errorbar(gain, mean_net_power, unc_gain_mean_net_power,
                  c='r', zorder=1, fmt="None")
    ax_1.scatter(gain, mean_net_power, c='#0033cc', zorder=2,
                 label=label, s=5)
    
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if SAVE_PLOTS:
        
        plt.savefig(IMAGE_DIRECTORY + model_name +
                    "_mean_net_power_against_gain",
                    dpi=600)
    
    plt.show()
    plt.close()
    

def main(tau_c_filename=TAU_C_FILENAME, d_ne_filename=D_NE_FILENAME,
         gain_filename=GAIN_FILENAME):
    
    if PLOT_CORRELATION_TIME_DATA:
        
        tau_c_gain, tau_c_model = identify_data_parameters(tau_c_filename)
        if tau_c_model == 1:
            tau_c_model_name = "AOUP"
        elif tau_c_model == 2:
            tau_c_model_name = "PLOUP"
            
        correlation_time_analysis_data = np.genfromtxt(DATA_DIRECTORY + tau_c_filename,
                                                delimiter=',', skip_header=1)
        
        
        tau_c_logspace = correlation_time_analysis_data[:,0]
        tau_c_mean_free_powers = correlation_time_analysis_data[:,1] 
        unc_tau_c_mean_free_powers = correlation_time_analysis_data[:,2]
        tau_c_mean_trap_powers = correlation_time_analysis_data[:,3]
        unc_tau_c_mean_trap_powers = correlation_time_analysis_data[:,4]
        tau_c_mean_net_powers = correlation_time_analysis_data[:,5]
        unc_tau_c_mean_net_powers = correlation_time_analysis_data[:,6]
        
        if CALCULATE_LOW_LIMITS:
            tau_c_low_power_limit = tau_c_non_active_bath_powers(TAU_C_D_NE, tau_c_gain,
                                                                 TAU_C_DELTA_G)
        else:
            tau_c_low_power_limit = LOW_TAU_C
        
        plot_free_power_against_correlation_time(tau_c_logspace,
                                                 tau_c_mean_free_powers,
                                                 unc_tau_c_mean_free_powers,
                                                 tau_c_low_power_limit,
                                                 tau_c_gain, tau_c_model_name,
                                                 TAU_C_D_NE, TAU_C_DELTA_G)
        
        plot_trap_power_against_correlation_time(tau_c_logspace,
                                                 tau_c_mean_trap_powers,
                                                 unc_tau_c_mean_trap_powers,
                                                 tau_c_gain, tau_c_model_name,
                                                 TAU_C_D_NE, TAU_C_DELTA_G)
        
        plot_net_power_against_correlation_time(tau_c_logspace,
                                                tau_c_mean_net_powers,
                                                unc_tau_c_mean_net_powers,
                                                tau_c_gain, tau_c_model_name,
                                                TAU_C_D_NE, TAU_C_DELTA_G)
    
    
    if PLOT_NOISE_STRENGTH_DATA:

        d_ne_gain, d_ne_model = identify_data_parameters(d_ne_filename)
        if d_ne_model == 1:
            d_ne_model_name = "AOUP"
        elif d_ne_model == 2:
            d_ne_model_name = "PLOUP"
        
        noise_strength_analysis_data = np.genfromtxt(DATA_DIRECTORY + d_ne_filename,
                                                      delimiter=',', skip_header=1)
        
        
        d_ne_logspace = noise_strength_analysis_data[:,0]
        d_ne_mean_free_powers = noise_strength_analysis_data[:,1] 
        unc_d_ne_mean_free_powers = noise_strength_analysis_data[:,2]
        d_ne_mean_trap_powers = noise_strength_analysis_data[:,3]
        unc_d_ne_mean_trap_powers = noise_strength_analysis_data[:,4]
        d_ne_mean_net_powers = noise_strength_analysis_data[:,5]
        unc_d_ne_mean_net_powers = noise_strength_analysis_data[:,6]
        
        if CALCULATE_LOW_LIMITS:
            d_ne_low_power_limit = d_ne_non_active_bath_powers(D_NE_TAU_C, d_ne_gain,
                                                               D_NE_DELTA_G)
        else:
            d_ne_low_power_limit = LOW_D_NE

        plot_free_power_against_noise_strength(d_ne_logspace,
                                               d_ne_mean_free_powers,
                                               unc_d_ne_mean_free_powers,
                                               d_ne_low_power_limit,
                                               d_ne_gain, d_ne_model_name,
                                               D_NE_TAU_C, D_NE_DELTA_G)
        
        plot_trap_power_against_noise_strength(d_ne_logspace,
                                               d_ne_mean_trap_powers,
                                               unc_d_ne_mean_trap_powers,
                                               d_ne_gain, d_ne_model_name,
                                               D_NE_TAU_C, D_NE_DELTA_G)
        
        plot_net_power_against_noise_strength(d_ne_logspace,
                                              d_ne_mean_net_powers,
                                              unc_d_ne_mean_net_powers,
                                              d_ne_gain, d_ne_model_name,
                                              D_NE_TAU_C, D_NE_DELTA_G)
        
    if PLOT_GAIN_DATA:
    
        gain_model = identify_data_parameters(gain_filename)
        if gain_model == 1:
            gain_model_name = "AOUP"
        elif gain_model == 2:
            gain_model_name = "PLOUP"
        
        feedback_gain_analysis_data = np.genfromtxt(DATA_DIRECTORY + gain_filename,
                                                      delimiter=',', skip_header=1)
        
        
        gain_linspace = feedback_gain_analysis_data[:,0]
        gain_mean_free_powers = feedback_gain_analysis_data[:,1] 
        unc_gain_mean_free_powers = feedback_gain_analysis_data[:,2]
        gain_mean_trap_powers = feedback_gain_analysis_data[:,3]
        unc_gain_mean_trap_powers = feedback_gain_analysis_data[:,4]
        gain_mean_net_powers = feedback_gain_analysis_data[:,5]
        unc_gain_mean_net_powers = feedback_gain_analysis_data[:,6]
    
        plot_free_power_against_gain(gain_linspace,
                                     gain_mean_free_powers,
                                     unc_gain_mean_free_powers,
                                     gain_model_name,
                                     GAIN_TAU_C, GAIN_D_NE,
                                     GAIN_DELTA_G)
        
        plot_trap_power_against_gain(gain_linspace,
                                     gain_mean_trap_powers,
                                     unc_gain_mean_trap_powers,
                                     gain_model_name,
                                     GAIN_TAU_C, GAIN_D_NE,
                                     GAIN_DELTA_G)
        
        plot_net_power_against_gain(gain_linspace,
                                    gain_mean_net_powers,
                                    unc_gain_mean_net_powers,
                                    gain_model_name,
                                    GAIN_TAU_C, GAIN_D_NE,
                                    GAIN_DELTA_G)
    
  
if __name__ == '__main__':
    main()
