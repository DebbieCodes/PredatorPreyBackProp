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
t_end = 40
dt = 0.05

# model parameters
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
    
    
measured_traj = traj_true*npr.normal(0.8, 1.1, traj_true.shape) # add noise
  
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

def predictStep2(params, inputs):
    new_state = RK4(LotkaVolterra,inputs,dt,params)
    return new_state

def loss(params):
    output = predictStep2(params, inputs)
    mse = np.sum((targets - output)**2)
    lambda_n = 1
    regularization =np.sum(lambda_n*(inputs-RK4(LotkaVolterra,inputs_prev,dt,params))) # make sure the step obeys the equations
    loss = mse + regularization
    return loss

#%% Inner loop: for one step optimize the paramters:
# assume we measure both the predators and the preys (i.e.e all states)

# guess paramters:    
alpha_g = alpha
beta_g = 0.9*beta
gamma_g = 0.9*gamma
delta_g = 0.9*delta
params_g = np.array([alpha_g, beta_g, gamma_g,  delta_g])
inputs = [x0,y0]
training_step = 0.01

print('untrained')
print(params_g)


#%% forward pass:
predicted_traj = []
grads_steps = []
lambda_steps = []
for step_i in range(nt-1):
    
    if step_i > 0:
        inputs_prev = measured_traj[step_i-1,:]
    inputs = measured_traj[step_i,:]
    targets = measured_traj[step_i+1,:]
    predictions = RK4(LotkaVolterra,inputs,dt,params_g)
    
    predicted_traj.append(predictions)
    grads_steps.append(grad(RK4,4)) # collect gradients

#%% Backward pass:
for step_k in range(nt,0,-1):    
    if step_k == nt:
        #here I am confused, therre are 2 states, so y_n-h(x_n) is not a scalar, but a 2 value vector..? So I do a sum...
        lambda_n= np.sum(predicted_traj[step_k,:]-measured_traj[step_k,:])
    elif step_k ==1:
        lamda_n=0 # ??
    else: 
        inputs=predicted_traj[step_k,:]
        lambda_n_plus1 = lambda_steps(step_k+1)
        lambda_n= np.sum(predicted_traj[step_k,:]-measured_traj[step_k,:]) + grad(RK4(LotkaVolterra,inputs,dt,params_g))*lambda_n_plus1
        
        
    lambda_steps.append(lambda_n)
  

training_gradient_fun = grad(loss)
training_gradient_fun(params_g)

for i in range(100):
    params_g -= training_gradient_fun(params_g)*training_step
    
# optimized_params = sgd(grad(loss), params_g, step_size=0.01, num_iters=5000)
    
print('trained')
print(params_g)
print(loss(params_g))

print('true')
print(params)
print(loss(params))
