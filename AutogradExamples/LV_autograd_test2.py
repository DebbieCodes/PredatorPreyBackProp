# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:48:33 2021

@author: eeltink
"""
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

def LotkaVolterra2(t,state): 
    x = np.array(state)
    n = len(x)
    f = np.zeros(n)
    print(beta)
    print(x[0])
    f[0] = alpha*x[0] - beta*x[0]*x[1] # xdot = alpha*x - beta*x*y
    f[1] = delta*x[0]*x[1] - gamma*x[1] # ydot = delta*x*y - gamma*y
    return f

Ca0 = 1.0
k1 = k_1 = 3.0

def dCdt(t, Ca):
    return -k1 * Ca + k_1 * (Ca0 - Ca)

t, Ca = rk4(dCdt, (0, 0.5), Ca0)

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
t, sols = rk4(LotkaVolterra2,(0, 20),u0)
sols=np.array(sols)
x_sol = sols[:,0]
y_sol = sols[:,1]

plt.plot(t, x_sol, label='state 1')
plt.plot(t, y_sol, label='state 2')
plt.xlabel('t')
plt.ylabel('[A]')
plt.legend()

def B(u0,alpha,beta,gamma,delta,t):
    def LotkaVolterra2(t,state):
        x = np.array(state)
        n = len(x)
        f = np.zeros(n)
        print(beta)
        print(x[0])
        f[0] = alpha*x[0] - beta*x[0]*x[1] # xdot = alpha*x - beta*x*y
        f[1] = delta*x[0]*x[1] - gamma*x[1] # ydot = delta*x*y - gamma*y
        return f
    t, sols_ = rk4(LotkaVolterra2,(0, t),u0)
    sols=np.array(sols)
    return sols_[-1,:]
        

def A(Ca0, k1, k_1, t):
    def dCdt(t, Ca):
        return -k1 * Ca + k_1 * (Ca0 - Ca)
    t, Ca_ = rk4(dCdt, (0, t), Ca0)
    return Ca_[-1]

# Here are the two derivatives we seek.
dCadk1 = grad(A, 1)
dCadk_1 = grad(A, 2)

der_alpha = grad(B, 1)
tspan = np.linspace(0, 0.5)



tspan = np.linspace(0, 0.5)

# From the numerical solutions
k1_sensitivity = [dCadk1(1.0, 3.0, 3.0, t) for t in tspan]
k_1_sensitivity = [dCadk_1(1.0, 3.0, 3.0, t) for t in tspan]
