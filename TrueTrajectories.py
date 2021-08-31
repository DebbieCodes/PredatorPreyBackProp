#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:27:49 2021
Lorentz96 example
@author: DebbieEeltink
"""

import numpy as np
import matplotlib.pyplot as plt

import models
#%% Settings
model = 'LV' #'Lorenz96'
t_end = 100
dt = 0.01


if model == 'Lorenz96':
    # settings
    F = 8 #forcing term
    n_states = 4
    u0 = models.lorenz96_IC(n_states,F)
    modelParameters = {'F': F} 
elif model =='LV':
    alpha = 1.1
    beta = 0.4
    gamma = 0.4
    delta = 0.1
    
    # Initial Values
    x0 = 20
    y0 = 5
    
    u0 = [x0,y0]
    modelParameters = {'alpha': alpha, 'beta': beta,'gamma': gamma,'delta': delta} 

else:
    print('No valid model')
    

# now run model for total time:
true_traj = models.runModelTot(u0,t_end,dt,modelParameters,model)
t = np.arange(0,t_end,dt)

#%% plot of 2 first states:
    
plt.figure()
plt.subplot(1,2,1)
plt.plot(t,true_traj[:,0], label='State 1 (prey)', linewidth=1)
plt.plot(t,true_traj[:,1], label='State 2 (predator)', linewidth=1)
plt.grid()
plt.legend(loc = 'upper right')
plt.xlabel('Time')

plt.subplot(1,2,2)
plt.plot(true_traj[:,0],true_traj[:,1], linewidth=1)
plt.grid()
plt.xlabel('State 1')
plt.ylabel('State 2')
plt.title('Phase Portrait')