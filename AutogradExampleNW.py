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


def sigmoid(x):
    return 0.5*(np.tanh(x)+1)

def logistic_predictions(weights, inputs):
    return sigmoid(np.dot(inputs,weights))

def training_loss(weights):
    # training loss is the negative log likelihood of training labels
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds*targets +  ( 1- preds)*(1-targets)

    return -np.sum(np.log(label_probabilities))

# load the data:
    
training_gradient_fun = grad(training_loss)

weights = np.array([0.0, 0.0, 0.0])
print("initial loss:", training_loss(weights))

training_step = 0.01
for i in range(100):
    weights -= training_gradient_fun(weights)*training_step


