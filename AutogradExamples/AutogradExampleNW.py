# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 18:19:08 2021
https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L06%20Automatic%20Differentiation.pdf
@author: eeltink
"""

from __future__ import absolute_import



import sys
if '../' not in sys.path:
  sys.path.append('../')
  
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import autograd.numpy.random as npr

def sigmoid(x):
    return 0.5*(np.tanh(x)+1)

def logistic_predictions(weights, inputs):
    return sigmoid(np.dot(inputs,weights))

# def training_loss(weights):
    
#     preds = logistic_predictions(weights, inputs)
#     # label_probabilities = preds*targets +  ( 1- preds)*(1-targets), # training loss is the negative log likelihood of training labels
    
#     return -np.sum(np.log(label_probabilities))

def training_loss(weights):
    
    preds = logistic_predictions(weights, inputs)
    loss = (preds-targets)**2
    
    return loss

    
    #%% Generate synthetic data
x = np.linspace(-5, 5, 1000)
t = x ** 3 - 20 * x + 10 + npr.normal(0, 4, x.shape[0])
plt.figure()
plt.plot(x, t, 'r.')

#%% NN
inputs = x.reshape(x.shape[-1],1)
targets = t.reshape(t.shape[-1],1)
    


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

def loss(params):
    output = predict(params, inputs)
    return (1.0 / inputs.shape[0]) * np.sum(0.5 * np.square(output.reshape(output.shape[0]) - t))

training_gradient_fun = grad(loss)
print("initial loss:", loss(params))
print(params)
training_step = 0.01
for i in range(100):
    params -= training_gradient_fun(params)*training_step


