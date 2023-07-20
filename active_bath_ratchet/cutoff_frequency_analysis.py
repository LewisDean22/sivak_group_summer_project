# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:52:20 2023

The distribution of free powers was so wide because my scaled effective mass
was way too small. By a factor of 10^12. Now the engine is more predictable,
less trajectories need averaging over.

RuntimeWarning: invalid value encountered in double_scalars - Calculations
involving too small or too large numbers. I think euler-maruyama was unstable
because delta_t was too large. Happens still for the larger f_ne values in
the logspace.

How to scale cut-off frequency? This has now been removed

USED T_PROTOCOL = 1 by accident! May want to repeat the code run for
T_PROTOCOL = 10.


@author: lewis
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from active_information_ratchet import calculate_mean_free_power
from model_functions import * #noqa
import constants_active as constants

IMAGE_DIRECTORY = "plots/"

F_NE_MIN_POWER = -2
F_NE_MAX_POWER = 4
N = 1 + (F_NE_MAX_POWER - F_NE_MIN_POWER)
D_NE = 3

PLOT_SAVE = 1


def plot_power_against_cutoff_frequency(f_ne, mean_free_power):

    fig, ax_1 = plt.subplots()
    # ax2 = ax1.twinx()
    label = (r"$D_{ne}$ = " + "{:.3g}".format(D_NE) + '\n' +
             r"$\delta_g$ = " + "{:.2g}".format(constants.DELTA_G))
    
    plt.title(r"Normalised $\langle \dot{F} \rangle$ as a function of $f_{ne}$",
              fontsize=18)
    ax_1.set_xlabel(r"$f_{ne}$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\frac{\langle \dot{F} \rangle}{\langle \dot{F_0} \rangle}$",
               rotation=0, labelpad=25, fontsize=18)
    ax_1.set_xscale("log")
    ax_1.scatter(f_ne, mean_free_power, c='r', zorder=1)
    ax_1.plot(f_ne, mean_free_power, c='#42b0f5', linestyle='-',
            zorder=0, label=label)
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if PLOT_SAVE:
        
        plt.savefig(IMAGE_DIRECTORY + "mean_free_power_against_cutoff_frequency",
                    dpi=600)
    
    plt.show()
    plt.close()
    

def main():
    '''
    Normalised vertical axis by the free power for the lowest noise strength.
    The power was in 10^-11 units originally so scaling broken somewhere.
    '''
    
    f_ne_logspace = np.logspace(F_NE_MIN_POWER, F_NE_MAX_POWER, num=12, base=10)
    
    mean_free_powers = []
    for f_ne in tqdm(f_ne_logspace):
        mean_free_powers.append(calculate_mean_free_power(f_ne, D_NE))

    plot_power_against_cutoff_frequency(f_ne_logspace,
                                        mean_free_powers/mean_free_powers[0])
    
  
if __name__ == '__main__':
    main()
