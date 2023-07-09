# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:57:42 2023

So this is a discrete-time, continuous-state space simulation.
Is it of physical interest to make this continuous-time? As in
sampling from some distribution (perhaps exponetial) for when
time steps should be taken, as opposed to a set number of
Euler-Maruyama iterations for a given time interval.

Seems like the standard deviation isn't sqrt(delta_t), this is for
the Wiener process alone. The factors that constitute the diffusion
term in the ou process have scaled this standard deviation by
sqrt(2D)/gamma. Or has it?? Some characteristic relaxation of the
system standard deviation can be seen.

Maybe a finer grid is needed to avoid errors here in more
complicated potentials?

Derivative failing near steep ratchet barrier?

An information ratchet on the other hand, may be a
quadratic potential well which has the location of its
minimum updated when measured so as to induce directed motion
from thermal fluctuations.

@author: lewis
"""

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import time as tm # noqa

IMAGE_DIRECTORY = "plots/"

SAMPLE_PATHS = 1
POTENTIAL_ANIMATION = 1 # Only for one example trajectory
PLOT_HEAT = 0
PLOT_PATHS = 0
PLOT_STATS = 0

PLOT_SAVE = 0

X_INIT = 5
N = 1000 # Number of time steps
TRAJECTORY_TIME = 1
DELTA_T = TRAJECTORY_TIME / N

V_MAX = 10
L = 1
DELTA = 0.05
CYCLE_FREQUENCY = 20
LINEAR_OPPOSING_FORCE = 0 # Magnitude

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
        dx_function = lambda x, t, cycle_num: (
            eval(self.drift) * delta_t + eval(self.diffusion) * delta_w)

        return dx_function


def gentle_slope(x,  n, d=DELTA, l=L, v_max=V_MAX):
    x_position = n * l
    prefactor = v_max/((1-d)*l)
    
    return -prefactor*(x-x_position) + v_max


def steep_slope(x,  n, d=DELTA, l=L, v_max=V_MAX):
    x_position = n*l + (1-d)*l
    prefactor = v_max/(d*l)
    
    return prefactor*(x-x_position)


def ratchet_potential_function(x, cycle_num, d=DELTA, l=L,
                                v_max=V_MAX, f=CYCLE_FREQUENCY,
                                opp_force=LINEAR_OPPOSING_FORCE):
    
    n_x = 0
    
    if (cycle_num % 2 == 0):
    
        for n in range(100):
            if n*l <= x < (n+1)*l:
                n_x = n
                break
        
        if x <=  (n_x*l + (1-d)*l):
            return gentle_slope(x, n_x)
        else:
            return steep_slope(x, n_x)
    
    else:
        return opp_force*x
    

def midpoint_derivative(x, cycle_num, step_size):
    h = step_size
    term_1 = ratchet_potential_function(x+h/2, cycle_num)
    term_2 = ratchet_potential_function(x-h/2, cycle_num)
    return (term_1 - term_2) / h


sawtooth = process("-midpoint_derivative(x, cycle_num, 0.0001)",
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
    cycle_num = [0]
    
    for n in range(N):
        x = x_n[-1]
        t = t_n[-1]
        
        dw = wiener_increment(dt)
        delta_x_n = sawtooth.delta_x(dt, dw)(x, t, cycle_num[-1])
        
        t_n += [(n+1)*dt]
        x_n += [x_n[-1] + delta_x_n]
        
        if t_n[-1] >= (cycle_num[-1]+1)/CYCLE_FREQUENCY:
            cycle_num += [cycle_num[-1] + 1]
        else:
            cycle_num += [cycle_num[-1]]
    
    return t_n, x_n, cycle_num


def sample_path_generator(i):
        
        return  euler_method(DELTA_T)


def generate_sawtooth(x, x_lim, d=DELTA, l=L):
    
    sawtooth = np.zeros_like(x)
    
    n = 0
    for count, x_val in enumerate(x):
        condition_1 = n*l + (1-d)*l
        condition_2 = (n+1)*l
        
        if x_val <= condition_1:
            sawtooth[count] = gentle_slope(x_val, n)
            continue
        if x_val <= condition_2:
            sawtooth[count] = steep_slope(x_val, n)
            continue
        
        sawtooth[count] = gentle_slope(x_val, n+1)
        n += 1
                   
    return sawtooth


def plot_potential(x_array, t_array, cycle_nums,
                   f=CYCLE_FREQUENCY,
                   opp_force=LINEAR_OPPOSING_FORCE):
        
    xlim_low = 0
    xlim_high = 10
    ylim_low = 0
    ylim_high = V_MAX + 1
    
    x_linspace = np.linspace(xlim_low, xlim_high, 10_000)
    sawtooth = generate_sawtooth(x_linspace, xlim_high)
    for count, x in enumerate(x_array):
        cycle_num = cycle_nums[count]
        
        plt.scatter(x, ratchet_potential_function(x, cycle_num),
                    c='r', zorder=1)
        
        
        if (cycle_num % 2 == 0):
            plt.plot(x_linspace, sawtooth, zorder=0)
        else:
            plt.plot(x_linspace, opp_force*x_linspace, zorder=0)
        
        
        plt.xlim((xlim_low, xlim_high))
        plt.ylim((ylim_low, ylim_high))
        plt.title('Brownian ratchet fluctuation {}'.format(count))
        plt.xlabel(r"$x$")
        plt.ylabel(r"$V(x)$")
        
        plt.show()
        plt.close()
        
        
def plot_trajectory_stats(time, mean_x,
                               std_x):
    '''
    '''
    
    fig_title = 'Statistics from {:,} Brownian ratchet trajectories'.format(SAMPLE_PATHS)
    plt.suptitle(fig_title, fontsize=15)
    
    plt.subplot(2,1,1)
    plt.plot(time, mean_x)
    plt.title(r'Mean $x$')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\langle x - x_{init} \rangle$')
    
    textbox = dict(boxstyle='round', edgecolor='k', facecolor='w',
                   alpha=0.75)

    box_label = (r'$\delta$ = {}'.format(DELTA) +
                 '\n' + r'$f$ = {} Hz'.format(CYCLE_FREQUENCY) +
                 '\n' + r'$V_{max}$ ' + '= {}'.format(V_MAX) +
                 '\n' + r'$D$ = {}'.format(process.D))
    
    plt.text(1.08, np.min(mean_x), box_label,
              fontsize=10, verticalalignment='bottom', bbox=textbox)
    
    plt.subplot(2,1,2)
    plt.plot(time, std_x)
    plt.title(r'Standard deviation in $x$')
    plt.xlabel(r'$t$ (s)')
    plt.ylabel(r'$\sigma$')
    
    plt.tight_layout()
    
    if PLOT_SAVE:
        plt.savefig(IMAGE_DIRECTORY + "Brownian_ratchet_trajectory_stats.png",
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
    cycle_numbers = trajectories[:,2]
    
    
    if POTENTIAL_ANIMATION:
        
        plot_potential(path_displacements[0], path_times[0],
                       cycle_numbers[0])
        
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
