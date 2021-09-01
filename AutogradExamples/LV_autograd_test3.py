# -*- coding: utf-8 -*-
"""
jj
"""
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def rk4(f, tspan, y0, alpha,beta,gamma,delta,N=50):
    x, h = np.linspace(*tspan, N, retstep=True)
    y = []
    y = y + [y0]
    for i in range(0, len(x) - 1):
        k1 = h * f(x[i], y[i],alpha,beta,gamma,delta)
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2,alpha,beta,gamma,delta)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2,alpha,beta,gamma,delta)
        k4 = h * f(x[i + 1], y[i] + k3,alpha,beta,gamma,delta)
        y += [y[-1] + (k1 + (2 * k2) + (2 * k3) + k4) / 6]
    return x, y

def LotkaVolterra2(t,state,alpha,beta,gamma,delta): 
    x = np.array(state)
    n = len(x)
    print(beta)
    print(x[0])
    f0 = alpha*x[0] - beta*x[0]*x[1] # xdot = alpha*x - beta*x*y
    f1 = delta*x[0]*x[1] - gamma*x[1] # ydot = delta*x*y - gamma*y
    output = np.array([f0, f1])
    return output

# Ca0 = 1.0
# k1 = k_1 = 3.0

# def dCdt(t, Ca):
#     return -k1 * Ca + k_1 * (Ca0 - Ca)

# t, Ca = rk4(dCdt, (0, 0.5), Ca0)


# params = { 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta}
# Initial Values
x0 = 20
y0 = 5
u0 = [x0,y0]
# t, sols = rk4(LotkaVolterra2,(0, 20),u0)
# sols=np.array(sols)
# x_sol = sols[:,0]
# y_sol = sols[:,1]

# plt.plot(t, x_sol, label='state 1')
# plt.plot(t, y_sol, label='state 2')
# plt.xlabel('t')
# plt.ylabel('[A]')
# plt.legend()

def B(u0,alpha,beta,gamma,delta,t):
    t, sols_ = rk4(LotkaVolterra2,(0, t),u0,alpha,beta,gamma,delta)
    sols_=np.array(sols_)
    return sols_[-1,0]
        


# Here are the two derivatives we seek.


der_alpha = grad(B, 1)



# From the numerical solutions
# model parameters
alpha0 = 1.1
beta0 = 0.4
gamma0 = 0.4
delta0 = 0.1
tspan = np.linspace(0, 20)

k1_sensitivity = [der_alpha(u0, alpha0, beta0, gamma0,delta0, t) for t in tspan]

