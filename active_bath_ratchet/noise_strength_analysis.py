# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:52:20 2023

The distribution of free powers was so wide because my scaled effective mass
was way too small. By a factor of 10^12. Now the engine is more predictable,
less trajectories need averaging over.

Need to change all constants such that the only thing which enters the code
which could be physical is the scaled effective mass.

@author: lewis
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.constants import g

from active_information_ratchet import calculate_mean_free_power
from model_functions import * #noqa
import constants_active as constants

IMAGE_DIRECTORY = "plots/"

D_NE_MIN_POWER = -1
D_NE_MAX_POWER = 2
N = 1 + (D_NE_MAX_POWER - D_NE_MIN_POWER)
F_NE = 118

PLOT_SAVE = 1


def plot_power_against_noise_strength(d_ne, mean_free_power):

    fig, ax = plt.subplots()
    label = (r"$f_{ne}$ = " + "{:.3g}".format(F_NE) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(constants.DELTA_G))
    
    plt.title(r"Normalised $\langle \dot{F} \rangle$ as a function of $D_{ne}$",
              fontsize=18)
    ax.set_xlabel(r"$D_{ne}$", labelpad=5, fontsize=18)
    ax.set_ylabel(r"$\frac{\langle \dot{F} \rangle}{\langle \dot{F_0} \rangle}$",
               rotation=0, labelpad=25, fontsize=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(d_ne, mean_free_power, c='r', zorder=1)
    ax.plot(d_ne, mean_free_power, c='#42b0f5', linestyle='-',
            zorder=0, label=label)
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if PLOT_SAVE:
        
        plt.savefig(IMAGE_DIRECTORY + "mean_free_power_against_noise_strength",
                    dpi=600)
    
    plt.show()
    plt.close()
    

def main():
    '''
    Normalised vertical axis by the free power for the lowest noise strength.
    The power was in 10^-11 units originally so scaling broken somewhere.
    '''
    
    d_ne_logspace = np.logspace(D_NE_MIN_POWER, D_NE_MAX_POWER, num=20, base=10)
    
    mean_free_powers = []
    for d_ne in tqdm(d_ne_logspace):
        mean_free_powers.append(calculate_mean_free_power(F_NE, d_ne))

    plot_power_against_noise_strength(d_ne_logspace,
                                      mean_free_powers/mean_free_powers[0])
    
  
if __name__ == '__main__':
    main()
