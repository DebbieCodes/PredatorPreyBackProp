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


def tanh(x):
    return (1.0 - np.exp(-x))  / (1.0 + np.exp(-x))

x = np.linspace(-7, 7, 200)
plt.plot(x, tanh(x),
         x, grad(tanh)(x),                                # first  derivative
         x, grad(grad(tanh))(x),                          # second derivative
         x, grad(grad(grad(tanh)))(x),                    # third  derivative
         x, grad(grad(grad(grad(tanh))))(x),              # fourth derivative
         x, grad(grad(grad(grad(grad(tanh)))))(x),        # fifth  derivative
         x, grad(grad(grad(grad(grad(grad(tanh))))))(x))  # sixth  derivative

plt.axis('off')
plt.savefig("tanh.png")
plt.show()