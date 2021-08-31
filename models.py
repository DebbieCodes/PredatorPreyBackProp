#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:27:49 2021
Lorentz96 example
@author: DebbieEeltink
"""

import numpy as np
import matplotlib.pyplot as plt
#%% Lorenz 96
def lorenz96_IC(n_states,F):
    
    sig_IC1 = 2
    x0 = np.random.normal(0,sig_IC1,[n_states,])   # Initial state 

    # time integration to remove transcients
    nt1 = 8192 
    dt = 0.04
    for k in range(nt1):
        x0True = RK4(Lorenz96,x0,dt,F)
    return x0

def RK4(rhs,state,dt,*args):
    
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2,*args)
    k3 = rhs(state+k2*dt/2,*args)
    k4 = rhs(state+k3*dt,*args)

    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state

def Lorenz96(state,*args): # Lorenz 96 model
    x = state
    F = args[0]
    n = len(x)    
    f = np.zeros(n)
    # bounday points: i=0,1,N-1
    f[0] = (x[1] - x[n-2]) * x[n-1] - x[0]
    f[1] = (x[2] - x[n-1]) * x[0] - x[1]
    f[n-1] = (x[0] - x[n-3]) * x[n-2] - x[n-1]
    for i in range(2, n-1):
        f[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
    # Add the forcing term
    f = f + F

    return f


def LotkaVolterra(state,*args): 
    x = state
    n = len(x)
    alpha = args[0]
    beta = args[1]    
    gamma = args[2]    
    delta = args[3]    
    f = np.zeros(n)
    
    f[0] = alpha*x[0] - beta*x[0]*x[1] # xdot = alpha*x - beta*x*y
    f[1] = delta*x[0]*x[1] - gamma*x[1] # ydot = delta*x*y - gamma*y
    return f

def runModelTot(IC, taulim,dtau, modelParameters, model):

    n_tau = int(taulim/dtau)
    n_states = len(IC)
    traj_true = np.zeros((n_tau,n_states))
    traj_true[0,:] = IC
    print(n_tau)
    print('n steps')
    
    if model == 'Lorenz96':
        F = modelParameters['F']
    
        for k in range(n_tau-1):
            k=k+1
            traj_true[k,:] = RK4(Lorenz96,traj_true[k-1,:],dtau,F)
            
    elif model == 'LV':
        alpha = modelParameters['alpha']
        beta = modelParameters['beta']
        gamma = modelParameters['gamma']
        delta = modelParameters['delta']
    
        for k in range(n_tau-1):
            k=k+1
            traj_true[k,:] = RK4(LotkaVolterra,traj_true[k-1,:],dtau,alpha,beta,gamma,delta)
        
    else:
        print('No valid model chosen')
        
    
    return traj_true

