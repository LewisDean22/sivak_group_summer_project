# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:57:42 2023

An information ratchet may be a quadratic potential well which
has the location of its minimum updated when particle position
measured so as to induce directed motion from thermal fluctuations.

This information ratchet continually updates. May be useful
to introduce a measuring frequency - then instantanoues update
here upon measurement.

@author: lewis
"""

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import time as tm # noqa

IMAGE_DIRECTORY = "plots/"

SAMPLE_PATHS = 10_000
POTENTIAL_ANIMATION = 0 # Only for one example trajectory

PLOT_STATS = 1
PLOT_SAVE = 1

X_INIT = 0
X_0 = 0
N = 1000 # Number of time steps
TRAJECTORY_TIME = 1
DELTA_T = TRAJECTORY_TIME / N

MEASURING_FREQUENCY = 100 #Hz

NUMBER_OF_CORES_USED = -1


class process:
    '''
    Free Brownian motion has no drift term. Only fluctuations
    drive the particle's trajectory.

    The Ornstein-Uhlenbeck process applies to overdamped
    Langevin dynamics within a quadratic potential. This then behaves
    as a simple harmonic system perturbed by some noise process.
    '''
    
    D = 10**3
    gamma = 1
    
    def __init__(self, drift_term, diffusion_term):
        self.drift = drift_term
        self.diffusion = diffusion_term

    
    def delta_x(self, delta_t, delta_w):
        dx_function = lambda x, x_0, t: (
            eval(self.drift) * delta_t + eval(self.diffusion) * delta_w)

        return dx_function


def trap_potential(x, x_0):
        
            return (x-x_0)**2
    

def drift_term(x, x_0):
    
        return -2*(x-x_0)
    

harmonic_trap = process("drift_term(x, x_0)",
                       "np.sqrt(2*process.D)")


def wiener_increment(dt):
    '''
    A Wiener increment is a gaussian random
    variable with mean zero and variance delta t.
    '''
    
    return np.random.normal(0, dt)


def euler_method(dt):
    '''
    Functions evaluated at the start of each time interval
    (Ito formalism)
    
    At the moment, the process type needs changing within the for
    loop manually
    '''
    
    t_n = [0]
    x_n = [X_INIT]
    x_0 = [X_0]
    
    for n in range(N):
        x = x_n[-1]
        t = t_n[-1]
             
        dw = wiener_increment(dt)
        delta_x_n = harmonic_trap.delta_x(dt, dw)(x, x_0[-1], t)
        
        t_n += [(n+1)*dt]
        x_n += [x_n[-1] + delta_x_n]
        
        if (t_n[-1]*MEASURING_FREQUENCY).is_integer() and x_n[-1] > x_0[-1]:
            x_0 += [x_n[-1]]
        else:
            x_0 += [x_0[-1]]
    
    return t_n, x_n, x_0


def sample_path_generator(i):
        
        return  euler_method(DELTA_T)


def plot_potential(x_array, t_array, well_positions):
        
    xlim = np.max(abs(x_array)) + 3
    
    x_linspace = np.linspace(-xlim, xlim, 10_000)
    for count, x in enumerate(x_array):
        x_0 = well_positions[count]
        
        plt.scatter(x, trap_potential(x, x_0),
                    c='r', zorder=1)

        plt.plot(x_linspace, trap_potential(x_linspace, x_0),
                 zorder=0)
        
        
        plt.xlim((-xlim, xlim))
        plt.title('Information ratchet fluctuation {}'.format(count))
        plt.xlabel(r"$x$")
        plt.ylabel(r"$V(x)$")
        
        plt.show()
        plt.close()


def plot_trajectory_stats(time, mean_x,
                               std_x):
    '''
    '''
     
    fig_title = 'Statistics from {:,} information ratchet trajectories'.format(SAMPLE_PATHS)
    plt.suptitle(fig_title, fontsize=15)
    
    plt.subplot(2,1,1)
    plt.plot(time, mean_x)
    plt.title(r'Mean $x$')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\langle x - x_{init} \rangle$')
    
    textbox = dict(boxstyle='round', edgecolor='k', facecolor='w',
                   alpha=0.75)

    box_label = (r'$f_{meas}$' + ' = {} Hz'.format(MEASURING_FREQUENCY) +
                 '\n' + r'$D$ = {}'.format(process.D))
    
    plt.text(1.08, np.max(mean_x) - 0.4, box_label,
              fontsize=10, verticalalignment='bottom', bbox=textbox)
    
    plt.subplot(2,1,2)
    plt.plot(time, std_x)
    plt.title(r'Standard deviation in $x$')
    plt.xlabel(r'$t$ (s)')
    plt.ylabel(r'$\sigma$')
    
    plt.tight_layout()
    
    if PLOT_SAVE:
        plt.savefig(IMAGE_DIRECTORY + "Information_ratchet_trajectory_stats.png",
                    dpi=600)
        
    plt.show()
    plt.close()


def main(n_cores=NUMBER_OF_CORES_USED):
    '''
    ddof=1 in np.std() uses the Bessel correction for sample
    variance.
    Mean displacement from intial position is generated here
    '''
           
    trajectories = Parallel(n_jobs=n_cores,
                            backend = "threading",
                            verbose=1)(
        delayed(sample_path_generator)(i) for i in range(SAMPLE_PATHS))
    
    trajectories = np.array(trajectories)
    path_times = trajectories[:,0]
    path_displacements = trajectories[:,1]
    well_positions = trajectories[:,2]
    
    if POTENTIAL_ANIMATION:
        
        plot_potential(path_displacements[0], path_times[0],
                       well_positions[0])
        
    if PLOT_STATS:
        mean_x = np.empty(N+1)
        std_x = np.empty(N+1)
        for count, x_t in enumerate(path_displacements.T):
            mean_x[count] = np.mean(x_t - X_INIT)
            std_x[count] = np.std(x_t, ddof=1)
        
        plot_trajectory_stats(path_times[0],
                              mean_x,
                              std_x)
    

if __name__ == '__main__':
    main()
