# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 18:35:54 2021
https://nbviewer.jupyter.org/url/www.cs.toronto.edu/~rgrosse/courses/csc421_2019/tutorials/tut2/autograd_tutorial.ipynb
@author: eeltink
Trying to implement: https://arxiv.org/pdf/2008.09915.pdf
equations 1-13
Should give same result as doing 'regular backprop' as in AUtogradExampleNW2
"""
import matplotlib.pyplot as plt
import autograd
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd import jacobian
from autograd.misc import flatten # flatten_func

from autograd.misc.optimizers import sgd

# def sgd(grad, init_params, callback=None, num_iters=200, step_size=0.1, mass=0.9):
#     """Stochastic gradient descent with momentum.
#     grad() must have signature grad(x, i), where i is the iteration number."""
#     flattened_grad, unflatten, x = flatten_func(grad, init_params)

#     velocity = np.zeros(len(x))
#     for i in range(num_iters):
#         g = flattened_grad(x, i)
#         if callback:
#             callback(unflatten(x), i, unflatten(g))
#         velocity = mass * velocity - (1.0 - mass) * g
#         x = x + step_size * velocity
#     return unflatten(x)

#%% Generate synthetic data
n_samples = 100
x = np.linspace(-5, 5, n_samples)
t = x ** 3 - 20 * x + 10 + npr.normal(0, 4, x.shape[0])
plt.figure()
plt.plot(x, t, 'r.')

#%% NN structure
inputs = x.reshape(x.shape[-1],1)
targets = t.reshape(t.shape[-1],1)

W1 = npr.randn(1,4)
b1 = npr.randn(4)
W2 = npr.randn(4,4)
b2 = npr.randn(4)
W3 = npr.randn(4,1)
b3 = npr.randn(1)
# alpha = paramters


params_tot =[[W1,b1],[W2,  b2],[W3, b3]]
params_weights = [W1,W2,W3]
params_biases = [b1,b2,b3]

# different options nonlinear function f_l can be different for different layers
def relu(x):
    return np.maximum(0, x)

nonlinearity = np.tanh
#nonlinearity = relu



# def my_z(params_weights_l,params_biases_l,input_vec):
#     params
#     z = np.dot(input_vec, params_weights_l) +params_biases_l
#     return z

def f_layer(params_l_flat,input_l,unflatten_func_l):
    params_l=unflatten_func_l(params_l_flat)
    weights = params_l[0]
    biases =  params_l[1]
    z= np.dot(input_l, weights) + biases
    output_l = nonlinearity(z)
    return output_l

def predict(params, inputs):
    h1 = nonlinearity(np.dot(inputs, params['W1']) + params['b1'])
    h2 = nonlinearity(np.dot(h1, params['W2']) + params['b2'])
    output = np.dot(h2, params['W3']) + params['b3']
    return output

def loss(params, i):
    output = predict(params, inputs)
    return (1.0 / inputs.shape[0]) * np.sum(0.5 * np.square(output.reshape(output.shape[0]) - t))



#%% Forward pass
n_layers=3 # number of layers
lambdas = list(np.zeros(n_layers+1))


for s in range(1):
    x_values = [] # store values each layer
    x0_sample = inputs[s][:,None]
    x_values.append(x0_sample)
    

    # 1. forward pass
    for l in range(0,n_layers,1):
        print("l =", l)
        params_l =params_tot[l]
        input_l = x_values[l]
        #have to flatten the paramters (i.e. make one long vector of all paramters) to calculate the jacobian later
        flat_params_l, unflatten_func_l = flatten(params_l)
        x_values.append(f_layer(flat_params_l,input_l,unflatten_func_l))
        # x_values.append(nonlinearity(np.dot(x_values[l-1], params_weights[l-1]) + params_biases[l-1]))
    
    # 2. terminal error:
    print(l+1)
    lambdas[l+1] = x_values[l+1]-targets[s] # recall there are 4 x_values but only 3 layers (parameter-valued)
    print(lambdas[3])
    # 3. backward pass
    # define jacobians: df/dx and df/dalpha, to be used later
    jacob_f_to_x = jacobian(f_layer,argnum=1) # i.e. jacobian of f to its 1st argument: x / state
    jacob_f_to_alpha = jacobian(f_layer,argnum=0) # i.e. jacobian of f to its 0th argument: alpha / parameters
    for k in range(n_layers-1,-1,-1):
        print("k =", k)
        flat_params_k, unflatten_func_k = flatten(params_tot[k])
        jac_notflat = jacob_f_to_x(flat_params_k,x_values[k],unflatten_func_k)
        jac_x_k = jac_notflat[0,:,0,:]
        # jac_x_k = np.squeeze(jac_notflat)#!!throws awy the wrong dimensiions
        lambda_k = jac_x_k.T*lambdas[k+1]
        lambdas[k] = lambda_k
        # lambdas[k] = 2*lambdas[k+1] # test if counter works correctly
        
    # 4. Parameter gradient (I guess this can also be done at the same time as the backward pass?)
    jac_J_alphas =[] # store jacobian of cost wrt to alpha
    for j in range(0,n_layers,1):
        print("j =", j)
        flat_params_j, unflatten_func_j = flatten(params_tot[j])
        jac_alpha_j = np.squeeze(jacob_f_to_alpha(flat_params_j,x_values[j],unflatten_func_j))
        print(jac_alpha_j.shape)
        print(lambdas[j+1].shape)
        jac_J_alpha = np.dot(jac_alpha_j.T,lambdas[j+1])
        jac_J_alphas.append(jac_J_alpha) # Jacobian of costwrt alpha
    
    

 
# optimized_params = sgd(grad(loss), params, step_size=0.01, num_iters=5000)
# print(optimized_params)
# print(loss(optimized_params, 0))

# final_y = predict(optimized_params, inputs)
# plt.figure()
# plt.plot(x, t, 'r.')
# plt.plot(x, final_y, 'b-')

#%%