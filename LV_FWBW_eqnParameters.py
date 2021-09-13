# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:53:05 2021

@author: eeltink
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:19:01 2021
see 
https://github.com/HIPS/autograd/issues/58

@author: DebbieEeltink
"""
import matplotlib.pyplot as plt
import autograd
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc import flatten # flatten_func
from autograd import jacobian


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
    alpha = params[0]
    beta = params[1]
    gamma = params[2] 
    delta = params[3]  
    
    f0 = alpha*x[0] - beta*x[0]*x[1] # xdot = alpha*x - beta*x*y
    f1 = delta*x[0]*x[1] - gamma*x[1] # ydot = delta*x*y - gamma*y
    output = np.array([f0, f1])
    return output



#%% Settings
t_end = 20
dt = 0.05

# true model parameters
alpha = 1.1
beta = 0.4
gamma = 0.4
delta = 0.1
# params = { 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta}
params = np.array([alpha, beta, gamma,  delta])
print('true')
print(params)
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
    
# measured trajectory not used for now (the true trajectories are used to calculate the loss)
traj_measured = traj_true*npr.normal(0.8, 1.1, traj_true.shape) # add noise
  
#%% Plots
fig, ax = plt.subplots(1,2)
ax[0].plot(t,traj_true[:,0], label='State 1 (True)',  color='orange', linewidth=1)
ax[0].plot(t,traj_true[:,1], label='State 2 (True)',  color='blue', linewidth=1)
ax[0].grid()
ax[0].legend(loc = 'upper right')
ax[0].set( xlabel ='Time', ylabel='States')

ax[1].plot(traj_true[:,0],traj_true[:,1], linewidth=1)
ax[1].grid()
ax[1].set(xlabel = 'State 1', ylabel='State 2')


#%% Defs loss


def loss(traj_pred, traj_true):
    diff_traj = (traj_pred-traj_true) # nt by n_states matrix
    loss_per_state = np.sum(diff_traj**2,axis=0) # sum over all times
    loss = np.sum(loss_per_state) # sum over all states
    
    return loss
    

#%% 
# assume we measure both the predators and the preys (i.e. all states)
# guess paramters that are not too far off from the true values (this is cheating of course):    
alpha_g = 0.95*alpha
beta_g = 0.9*beta
gamma_g = 0.95*gamma
delta_g = 1.05*delta
params_g = np.array([alpha_g, beta_g, gamma_g,  delta_g])
inputs = [x0,y0]
training_step = 0.01

print('untrained')
print(params_g)

# generate trajectory untrained paramters:
traj_untrained = np.zeros((nt,n_states))
traj_untrained[0,:] = u0

for k in range(nt-1):
    k=k+1
    traj_untrained[k,:] = RK4(LotkaVolterra,traj_untrained[k-1,:],dt,params_g)
   
ax[0].plot(t,traj_untrained[:,0], label='State 1 (Untrained)', color='orange', linewidth=1, linestyle =':')
ax[0].plot(t,traj_untrained[:,1], label='State 2 (Untrained)', color='blue',linewidth=1, linestyle =':')

ax[1].plot(traj_untrained[:,0],traj_true[:,1], color='orange', linewidth=1, linestyle =':')


#%% Joint state and parameter estimation:
n_iterations = 50
# we do not loop over samples because at each timestep there is only one measurement (of the 2 states)
# define jacobians: df/dx and df/dalpha, to be used later
jacob_f_to_x = jacobian(RK4,argnum=1) # i.e. jacobian of f to its 1st argument: x / state
jacob_f_to_alpha = jacobian(RK4,argnum=3) # i.e. jacobian of f to its 0th argument: alpha / parameters
parameter_record = np.zeros((n_iterations,4))
loss_record = np.zeros(n_iterations)
for train_i in range(n_iterations):
    #%% 1. Forward pass:
    traj_pred = np.zeros_like(traj_true)
    traj_pred[0] = u0
    lambdas = list(np.zeros(nt))
    
    for step_l in range(1,nt,1):
        # print(step_l)
        inputs = traj_pred[step_l-1,:]
        traj_pred[step_l] = RK4(LotkaVolterra,inputs,dt,params_g)
    
    
    # %%2. terminal error:
    lambdas[step_l] = traj_pred[step_l]-traj_true[step_l] 
    # print(lambdas[step_l])
    #%% 3. backward pass
    
    for step_k in range(nt-2,-1,-1): # eq 12
        # print("k =", step_k)
        jac_x_k = jacob_f_to_x(LotkaVolterra,traj_pred[step_k,:],dt,params_g)
        lambda_k = np.dot(jac_x_k.T,lambdas[step_k+1])
        lambdas[step_k] = lambda_k
                
    #%% 4. Parameter gradient (I guess this can also be done at the same time as the backward pass?)
    jac_J_alphas = np.zeros((nt-1,4)) # store jacobian of cost wrt to alpha/the paramters (eq 13)
    for step_j in range(0,nt-1,1):
        # print("j =", step_j)
        jac_alpha_j = jacob_f_to_alpha(LotkaVolterra,traj_pred[step_j,:],dt,params_g) # Jac of function wrt alpha
        jac_J_alphas [step_j,:] = np.dot(jac_alpha_j.T,lambdas[step_j+1])  # Jacobian of cost wrt alpha
    
    #%% 5. Move parameters in direction of gradient: equation 14. 
    learning_rate = 0.0005
    params_upd = learning_rate*np.sum(jac_J_alphas,axis=0)
    params_g = params_g - params_upd
    # if parameters are negative, set to 0
    params_g[params_g < 0.01] = 0.01
    print('paramters training iteration', train_i+1)
    print(params_g)
    
    loss_record[train_i] = loss(traj_pred, traj_true)
    print('Loss: ', loss_record[train_i])
    parameter_record[train_i,:] = params_g
    
#%% Plots results:
    
fig2, ax2= plt.subplots(1,1)
ax2.semilogy(np.arange(n_iterations),loss_record)
ax2.set(xlabel='iteration', ylabel='loss')
    
#%% comparison figure:
fig, ax = plt.subplots(1,2)
ax[0].plot(t,traj_true[:,0], label='Prey (True)',  color='orange', linewidth=1)
ax[0].plot(t,traj_untrained[:,0], label='Prey (Untrained)', color='orange', linewidth=1, linestyle =':')
ax[0].plot(t,traj_pred[:,0], label='Prey (Trained)', color='orange', linewidth=1, linestyle ='--')
ax[0].plot(t,traj_true[:,1], label='Predator (True)',  color='blue', linewidth=1)
ax[0].plot(t,traj_untrained[:,1], label='Predator (Untrained)', color='blue',linewidth=1, linestyle =':')
ax[0].plot(t,traj_pred[:,1], label='Predator (Trained)', color='blue',linewidth=1, linestyle ='--')

ax[0].grid()
ax[0].legend(loc = 'upper right')
ax[0].set( xlabel ='Time', ylabel='States')

ax[1].plot(traj_true[:,0],traj_true[:,1], color='green', linewidth=1)
ax[1].plot(traj_untrained[:,0],traj_true[:,1], color='green', linewidth=1, linestyle =':')
ax[1].plot(traj_pred[:,0],traj_pred[:,1], color='green', linewidth=1, linestyle ='--')
ax[1].grid()
ax[1].set(xlabel = 'Prey', ylabel='Predator')
ax[1].legend()

