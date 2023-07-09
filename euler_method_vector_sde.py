# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:36:27 2023

NOTE: This is a simplified example in which N=M=2
N = number of dimensions and M = number of
independent Wiener processes. I.e. this would
apply to a particle able to move in two dimensions
and exposed to do independent sources of Gaussian
noise. 2D Brownian motion? Or does Brownian motion
have fixed displacement size per time interval?

Try to find the 2 dimensional standard deviation and mean!

Now how does this relate to the Langevin equation. That
equation is written in terms of derivatives whereas these
numerical methods act on SDEs written in terms of differentials.
One could try and rewrite the Langevin equation in terms of
such differentials - fdt and gdw = dx. Does this even make
sense? I read that the Ornstein-uhlenbeck process is a
stationary state of the Langevin equation.

@author: lewis
"""

import numpy as np
import matplotlib.pyplot as plt
import time as tm


SAMPLE_PATHS = 2
PLOT_PATHS = 1
PLOT_HIST = 0
DIMENSIONS = 2

VECTOR_X_INIT = np.array([0,0])
N = 1000 # Number of steps
TRAJECTORY_TIME = 1
DELTA_T = TRAJECTORY_TIME / N



def vector_f(x,t):
    '''
    If f is proportional to -x, the trajectory
    will correspond to that of SHM in the presence
    of Gaussian noise. The greater the constant of
    proportionality, the greater the tendency to
    return to x=0.
    '''
    x1 = x[0]
    x2 = x[1]
    
    # return np.array([-1*x1 -1*x2])
    return np.array([1, 1])


def matrix_g(x,t):
    '''
    An example of an Ornstein-Uhlenbeck process
    is an SDE in which the noise is "additive",
    i.e. it does not depend on the dependent
    variable (x).
    '''
    
    return np.array([[1,1], [1,1]])*1


def vector_wiener_increment():
    '''
    A Wiener increment is a gaussian random
    variable with mean zero and variance delta t.
    
    Method for producing similiar sample paths through
    2N gaussian Random numbers shown in Jacobs.
    '''
    dw1 = np.random.normal(0, DELTA_T)
    dw2 = np.random.normal(0, DELTA_T)
    
    return np.array([dw1, dw2])
    
    
def find_vector_delta_x(x,t):
    delta_w = vector_wiener_increment()
    return vector_f(x,t)*DELTA_T + np.dot(matrix_g(x,t), delta_w)


def euler_method():
    '''
    t_n = scalar
    x_n = vector
    '''
    
    t_n = [0] # evaluated at start of interval
    x_n = np.array([VECTOR_X_INIT])
    
    for n in range(N):
        # print(x_n[-1])
        delta_x_n = find_vector_delta_x(x_n[-1], t_n[-1])
        
        t_n += [(n+1)*DELTA_T]
        x_n = np.vstack((x_n, x_n[-1] + delta_x_n))
        
    
    return t_n, x_n
    
def plot_sample_path(t_array, x_array):
    
    
    fig = plt.figure(figsize=(7, 7))
    axes = plt.axes(projection='3d')
    axes.set_title(r"Euler method for 2D sample paths", c='r')
    axes.set_xlabel('\n\n' + r'$x$', fontsize=12,
                    c='#cc0000')
    plt.xticks(fontsize=8, rotation=-70)
    axes.set_ylabel('\n\n' + r'$y$', fontsize=12,
                    c='#cc0000')
    plt.yticks(fontsize=8, rotation=20)
    axes.set_zlabel('Time', fontsize=10, c='#cc0000')
    axes.view_init(30, 60)
    
    axes.scatter(0,0, color='r')
    
    for i in range(SAMPLE_PATHS):
        t = t_array[i]
        # print(x_array[i])
        x = x_array[i,:,0]
        y = x_array[i,:,1]

        axes.plot(x, y, t)
        
    plt.show()
    plt.close()
    
    return None


def main():
    
    path_times = np.zeros([SAMPLE_PATHS, N+1])
    path_displacements = np.zeros([SAMPLE_PATHS, N+1, DIMENSIONS])
    
    for i in range(SAMPLE_PATHS):
        
        time, displacement = euler_method()
        path_times[i,:] = time
        path_displacements[i,:] = displacement
    
    if PLOT_PATHS:
        plot_sample_path(path_times, path_displacements)
    
    mean_x = np.mean(path_displacements[:,-1])
    std_x = np.std(path_displacements[:,-1], ddof=1)
    # print("Mean x at t={0} is {1}.".format(TRAJECTORY_TIME, mean_x))
    # print("Standard deviation in x at t={0} is {1}.".format(TRAJECTORY_TIME,
    #                                                         std_x))
    
    return None

if __name__ == '__main__':
    main()
