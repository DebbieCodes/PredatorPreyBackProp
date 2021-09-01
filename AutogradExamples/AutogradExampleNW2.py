# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 18:35:54 2021
https://nbviewer.jupyter.org/url/www.cs.toronto.edu/~rgrosse/courses/csc421_2019/tutorials/tut2/autograd_tutorial.ipynb
@author: eeltink
"""
import matplotlib.pyplot as plt
import autograd
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
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
x = np.linspace(-5, 5, 1)
t = x ** 3 - 20 * x + 10 + npr.normal(0, 4, x.shape[0])
plt.figure()
plt.plot(x, t, 'r.')

#%% NN
inputs = x.reshape(x.shape[-1],1)
W1 = npr.randn(1,4)
b1 = npr.randn(4)
W2 = npr.randn(4,4)
b2 = npr.randn(4)
W3 = npr.randn(4,1)
b3 = npr.randn(1)

params = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3 }

def relu(x):
    return np.maximum(0, x)

nonlinearity = np.tanh
#nonlinearity = relu

def predict(params, inputs):
    h1 = nonlinearity(np.dot(inputs, params['W1']) + params['b1'])
    h2 = nonlinearity(np.dot(h1, params['W2']) + params['b2'])
    output = np.dot(h2, params['W3']) + params['b3']
    return output

def loss(params, i):
    output = predict(params, inputs)
    return (1.0 / inputs.shape[0]) * np.sum(0.5 * np.square(output.reshape(output.shape[0]) - t))

print(loss(params, 0))

optimized_params = sgd(grad(loss), params, step_size=0.01, num_iters=5000)
print(optimized_params)
print(loss(optimized_params, 0))

final_y = predict(optimized_params, inputs)
plt.figure()
plt.plot(x, t, 'r.')
plt.plot(x, final_y, 'b-')

#%%