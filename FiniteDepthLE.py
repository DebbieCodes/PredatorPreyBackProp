# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 21:00:21 2022

@author: eeltink
"""

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


#%% Generate synthetic data
n_samples = 100
learning_rate = 0.05
n_trainsteps = 12 # same results with 1000 training iterations
x = np.linspace(-5, 5, n_samples)
t = x ** 3 - 20 * x + 10 + npr.normal(0, 4, x.shape[0])
plt.figure()
plt.plot(x, t, 'r.')

#%% Network functions

# different options nonlinear function f_l can be different for different layers
def relu(x):
    return np.maximum(0, x)

nonlinearity = np.tanh
#nonlinearity = relu

def linear_activation(x):
    return x


# def my_z(params_weights_l,params_biases_l,input_vec):
#     params
#     z = np.dot(input_vec, params_weights_l) +params_biases_l
#     return z

def f_layer(params_l_flat,input_l,unflatten_func_l,activation_function):
    params_l=unflatten_func_l(params_l_flat)
    weights = params_l[0]
    biases =  params_l[1]
    z= np.dot(input_l, weights) + biases
    output_l = activation_function(z)
    return output_l



def predict2(params,inputs):
    flat_params_l1, unflatten_func_l1 = flatten(params[0])
    flat_params_l2, unflatten_func_l2 = flatten(params[1]) 
    flat_params_l3, unflatten_func_l3 = flatten(params[2]) 
    h1 = f_layer(flat_params_l1,inputs,unflatten_func_l1,activation_layers[0])
    h2 = f_layer(flat_params_l2,h1, unflatten_func_l2,activation_layers[1])
    output= f_layer(flat_params_l3,h2, unflatten_func_l3,activation_layers[2])
    return output

def predict(params, inputs):
    h1 = nonlinearity(np.dot(inputs, params['W1']) + params['b1'])
    h2 = nonlinearity(np.dot(h1, params['W2']) + params['b2'])
    output = np.dot(h2, params['W3']) + params['b3']
    return output

def loss(params, i):
    output = predict(params, inputs)
    return (1.0 / inputs.shape[0]) * np.sum(0.5 * np.square(output.reshape(output.shape[0]) - t))

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
params_init = params_tot.copy()

activation_layers = [np.tanh,np.tanh,linear_activation]
# n_layers = np.len(activation_layers) + 1 #input
n_layers = 3 + 1 #inpu
#%% Training

G_store = np.zeros((n_trainsteps,n_layers-1))

for tt in range(n_trainsteps):
    print(r'trainingstep {:0f}'.format(tt))
    n_layers=4 # number of node-layers (4), which means there are 3 weights matrixes (i.e. in between connections)
    lambdas = list(np.zeros(n_layers+1))
    jac_J_alphas_samples = []
    Eigs_it_store =[]
    for s in range(n_samples):
        Eigs_store = []
        # print("sample =", s)
        x_values = [] # store values each layer
        x0_sample = inputs[s][:,None]
        x_values.append(x0_sample)
        
    
        # %%1. forward pass
        for l in range(0,n_layers-1,1):
            # print("l =", l)
            params_l =params_tot[l]
            input_l = x_values[l]

            #have to flatten the paramters (i.e. make one long vector of all paramters) to calculate the jacobian later
            flat_params_l, unflatten_func_l = flatten(params_l)
            x_values.append(f_layer(flat_params_l,input_l,unflatten_func_l,activation_layers[l]))
            # x_values.append(nonlinearity(np.dot(x_values[l-1], params_weights[l-1]) + params_biases[l-1]))
        
        # %%2. terminal error:
        # print(l+1)
        lambdas[l+1] = x_values[l+1]-targets[s] 
        # print(lambdas[3])
        #%% 3. backward pass
        # define jacobians: df/dx and df/dalpha, to be used later
        jacob_f_to_x = jacobian(f_layer,argnum=1) # i.e. jacobian of f to its 1st argument: x / state
        jacob_f_to_alpha = jacobian(f_layer,argnum=0) # i.e. jacobian of f to its 0th argument: alpha / parameters
        for k in range(n_layers-2,-1,-1): # eq 12
            # print("k =", k)
            flat_params_k, unflatten_func_k = flatten(params_tot[k])
            jac_notflat = jacob_f_to_x(flat_params_k,x_values[k],unflatten_func_k,activation_layers[k])
            jac_x_k = jac_notflat[0,:,0,:]  # throw away extra dimension
            # jac_x_k = np.squeeze(jac_notflat)#!!throws awy the wrong dimensiions
            B = jac_x_k.T #[n_nodes,1]
            lambda_k = np.dot( B ,lambdas[k+1]) # [n_nodes,1]
            # in the text these are tildes
            C =  B@B.T
            U, S, Vh = np.linalg.svd(C, full_matrices=True)
            Sigma2 = np.diag(S)
            Eigs_store.append(S)
            G = np.sum(S)
            G_store[tt,k] = G
            
            
            # Lyupanov exponent calculation:
            # lambdaNorm = 
            
            lambdas[k] = lambda_k
            # lambdas[k] = 2*lambdas[k+1] # test if counter works correctly
            
        #%% 4. Parameter gradient (I guess this can also be done at the same time as the backward pass?)
        jac_J_alphas =[] # store jacobian of cost wrt to alpha (eq 13)
        Eigs_it_store.append(Eigs_store)
        for j in range(0,n_layers-1,1):
            # print("j =", j)
            flat_params_j, unflatten_func_j = flatten(params_tot[j])
            jac_alpha_j_notFlat = jacob_f_to_alpha(flat_params_j,x_values[j],unflatten_func_j,activation_layers[j])
            # jac_alpha_j = np.squeeze(jacob_f_to_alpha(flat_params_j,x_values[j],unflatten_func_j)) # does not get axes right
            jac_alpha_j = jac_alpha_j_notFlat[0,:,:]
            # print(jac_alpha_j.shape)
            # print(lambdas[j+1].shape)
            jac_J_alpha = np.dot(jac_alpha_j.T,lambdas[j+1])
            jac_J_alphas.append(jac_J_alpha) # Jacobian of costwrt alpha
        jac_J_alphas_samples.append(jac_J_alphas) # not very pretty coding this nested list
    #%% 5. Move parameters in direction of gradient: equation 14. Because of the nested list of the gradients this looks a bit clumsy. 
    for kk in range(n_layers-1):
        # print("layer weights=", kk)
        flat_params_kk, unflatten_func_kk = flatten(params_tot[kk])
        grad_params = [i[kk] for i in jac_J_alphas_samples]
        grad_params = np.squeeze(np.array(grad_params))
        avg_grad = (1/s)*np.sum(grad_params,0)
        flat_params_kk_upd = flat_params_kk - learning_rate*avg_grad
        params_kk_upd = unflatten_func_kk(flat_params_kk_upd)
        # store updated parameters:
        params_tot[kk] = params_kk_upd
    
    
#%% Compare intial and optimized:
# init_loss = loss(params_init, 0)
inputs = x.reshape(x.shape[-1],1)
pred_init = predict2(params_init, inputs)
pred_trained = predict2(params_tot, inputs)

plt.figure()
plt.plot(x, t, 'r.', label='data')
plt.plot(x,pred_init, label='initial parameters')
plt.plot(x,pred_trained, label ='trained parameters')
plt.legend()

#%% Plot norm stability criterion

fig,ax= plt.subplots(1,2)
for j in range(n_trainsteps):
    ax[0].plot(np.arange(n_layers-1),G_store[j,:],'k',alpha=((j+1)/n_trainsteps)*0.8)
ax[0].set(xlabel='layer k', ylabel ='G(k)',title='Sum of eigenvalues')
    
for jj in range(n_layers-1):
    eigs_plot=Eigs_it_store[-1][jj]
    ax[1].plot(jj,eigs_plot[:,None].T,'*')
ax[1].set(xlabel='layer k', ylabel =r'Eigvs $\Sigma$', title='Last iteration only')
