# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 12:37:17 2021

@author: eeltink
"""
import autograd.numpy as np
from autograd import grad
%matplotlib inline
import matplotlib.pyplot as plt

def rk4(f, tspan, y0, N=50):
    x, h = np.linspace(*tspan, N, retstep=True)
    y = []
    y = y + [y0]
    for i in range(0, len(x) - 1):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i + 1], y[i] + k3)
        y += [y[-1] + (k1 + (2 * k2) + (2 * k3) + k4) / 6]
    return x, y


Ca0 = 1.0
k1 = k_1 = 3.0

def dCdt(t, Ca):
    return -k1 * Ca + k_1 * (Ca0 - Ca)

t, Ca = rk4(dCdt, (0, 0.5), Ca0)

def analytical_A(t, k1, k_1):
    return Ca0 / (k1 + k_1) * (k1 * np.exp(-(k1 + k_1) * t) + k_1)

plt.plot(t, Ca, label='RK4')
plt.plot(t, analytical_A(t, k1, k_1), 'r--', label='analytical')
plt.xlabel('t')
plt.ylabel('[A]')
plt.xlim([0, 0.5])
plt.ylim([0.5, 1])
plt.legend()

def A(Ca0, k1, k_1, t):
    def dCdt(t, Ca):
        return -k1 * Ca + k_1 * (Ca0 - Ca)
    t, Ca_ = rk4(dCdt, (0, t), Ca0)
    return Ca_[-1]

# Here are the two derivatives we seek.
dCadk1 = grad(A, 1)
dCadk_1 = grad(A, 2)


dAdk1 = grad(analytical_A, 1)
dAdk_1 = grad(analytical_A, 2)

# From the numerical solutions
tspan = np.linspace(0, 0.5)
k1_sensitivity = [dCadk1(1.0, 3.0, 3.0, t) for t in tspan]
k_1_sensitivity = [dCadk_1(1.0, 3.0, 3.0, t) for t in tspan]



