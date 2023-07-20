# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:52:20 2023

NEED TO ADD UNITS TO VERTICAL AXIS

@author: lewis
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from free_power_calculator import calculate_mean_free_power


IMAGE_DIRECTORY = "plots/"
PLOT_SAVE = 0

DELTA_G_MIN = 0
DELTA_G_MAX = 3
NUMBER_OF_POINTS = 12

D_NE = 0.6
F_NE = 10**3


def plot_power_against_delta_g(delta_g, mean_free_power):

    fig, ax_1 = plt.subplots()
    # ax2 = ax1.twinx()
    label = (r"$D_{ne}$ = " + "{:.3g}".format(D_NE) + '\n' +
             r"$f_{ne}$ = " + "{:,}".format(F_NE))
    
    plt.title(r"$\langle \dot{F} \rangle$ as a function of $\delta_g$",
              fontsize=18)
    ax_1.set_xlabel(r"$\delta_g$", labelpad=5, fontsize=18)
    ax_1.set_ylabel(r"$\langle \dot{F} \rangle$",
               rotation=0, labelpad=25, fontsize=18)
    ax_1.set_ylim((0, np.max(mean_free_power)*1.5))
    ax_1.scatter(delta_g, mean_free_power, c='r', zorder=1)
    ax_1.plot(delta_g, mean_free_power, c='#42b0f5', linestyle='-',
            zorder=0, label=label)
    plt.legend(fontsize="12")
    plt.tight_layout() 
    
    if PLOT_SAVE:
        
        plt.savefig(IMAGE_DIRECTORY + "mean_free_power_against_delta_g",
                    dpi=600)
    
    plt.show()
    plt.close()
    

def main():

    
    delta_g_linspace = np.linspace(DELTA_G_MIN, DELTA_G_MAX,
                                   NUMBER_OF_POINTS)
    
    mean_free_powers = []
    for delta_g in tqdm(delta_g_linspace):
        
        mean_free_powers.append(calculate_mean_free_power(F_NE, D_NE,
                                                          delta_g))

    plot_power_against_delta_g(delta_g_linspace,
                               mean_free_powers)
    
  
if __name__ == '__main__':
    main()
