#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:19:01 2021

@author: DebbieEeltink
"""
import matplotlib.pyplot as plt
import autograd
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc import flatten # flatten_func

from autograd.misc.optimizers import sgd

def RK4(rhs,state,dt,params):
    
    k1 = rhs(state,params)
    k2 = rhs(state+k1*dt/2,params)
    k3 = rhs(state+k2*dt/2,params)
    k4 = rhs(state+k3*dt,params)

    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state

def LotkaVolterra(state,params): 
    x = state
    n = len(x)
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma'] 
    delta = params['delta']  
    f = np.zeros(n)
    
    f[0] = alpha*x[0] - beta*x[0]*x[1] # xdot = alpha*x - beta*x*y
    f[1] = delta*x[0]*x[1] - gamma*x[1] # ydot = delta*x*y - gamma*y
    return f

def LotkaVolterra2(state,alpha,beta,gamma,delta): 
    x = np.array(state)
    n = len(x)
    f = np.zeros(n)
    print(beta)
    print(x[0])
    f[0] = alpha*x[0] - beta*x[0]*x[1] # xdot = alpha*x - beta*x*y

    f[1] = delta*x[0]*x[1] - gamma*x[1] # ydot = delta*x*y - gamma*y
    return f

#%% Settings
t_end = 100
dt = 0.01

# model parameters
alpha = 1.1
beta = 0.4
gamma = 0.4
delta = 0.1
params = { 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta}
# Initial Values
x0 = 20
y0 = 5
u0 = [x0,y0]

#%% Generate synthetic data
n_states = len(u0)
nt= int(t_end/dt)
traj_true = np.zeros((nt,n_states))
traj_true[0,:] = u0
t = np.arange(0,t_end,dt)


for k in range(nt-1):
    k=k+1
    traj_true[k,:] = RK4(LotkaVolterra,traj_true[k-1,:],dt,params)
    
    
measured_traj = traj_true + npr.normal(0, 4, traj_true.shape) # add noise
  
#%% Plots
plt.figure()
plt.subplot(1,2,1)
plt.plot(t,traj_true[:,0], label='State 1 (prey)', linewidth=1)
plt.plot(t,traj_true[:,1], label='State 2 (predator)', linewidth=1)
plt.grid()
plt.legend(loc = 'upper right')
plt.xlabel('Time')

plt.subplot(1,2,2)
plt.plot(traj_true[:,0],traj_true[:,1], linewidth=1)
plt.grid()
plt.xlabel('State 1')
plt.ylabel('State 2')

#%% Initial guess paramaters (close to true value)





def predictStep(params, inputs): #input is previous step, # output is next step
    output = RK4(LotkaVolterra,inputs,dt,params)
    return output

def predictStep2(params, inputs):
    
    state = inputs
    rhs = LotkaVolterra2
    
    alpha = params[0]
    beta = params[1]
    gamma = params[2] 
    delta = params[3]
    
    k1 = rhs(state,alpha,beta,gamma,delta)
    k2 = rhs(state+k1*dt/2,alpha,beta,gamma,delta)
    k3 = rhs(state+k2*dt/2,alpha,beta,gamma,delta)
    k4 = rhs(state+k3*dt,alpha,beta,gamma,delta)

    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state

def loss(params):
    output = predictStep2(params, inputs)
    
    return np.sum((targets - output)**2)

#%% Inner loop: for one step optimize the paramters:
    # assume we measure both the predators and the preys (i.e.e all states)
    
alpha_g = alpha
beta_g = 0.9*beta
gamma_g = 0.9*gamma
delta_g = 0.9*delta
params_g = np.array([alpha_g, beta_g, gamma_g,  delta_g])

step_i = 2
inputs = measured_traj[step_i,:]
targets = measured_traj[step_i+1,:]

print(params_g)

print(loss(params_g))


training_step = 0.01
training_gradient_fun = grad(loss)
training_gradient_fun(params_g)
for i in range(100):
    params_g -= training_gradient_fun(params_g)*training_step
    
# optimized_params = sgd(grad(loss), params_g, step_size=0.01, num_iters=5000)
    

