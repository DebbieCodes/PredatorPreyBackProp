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



# Generate synthetic data
N = 100 # Number of data points
x = np.linspace(-3, 3, N) # Generate N values linearly-spaced between -3 and 3
t = x ** 4 - 10 * x ** 2 + 10 * x + npr.normal(0, 4, x.shape[0]) # Generate corresponding targets

plt.figure()
plt.plot(x, t, 'r.') # Plot data points

M = 4 # Degree of polynomial to fit to the data (this is a hyperparameter)
feature_matrix = np.array([[item ** i for i in range(M+1)] for item in x]) # Construct a feature matrix: 
    # first entry is 1, second is x, third is x^2 fourth is x^3
W = npr.randn(feature_matrix.shape[-1]) # parameters of the polynomial

def cost(W):
    y = np.dot(feature_matrix, W)
    return (1.0 / N) * np.sum(0.5 * np.square(y - t))

# cost before training
print(W)
print(cost(W))

# Compute the gradient of the cost function using Autograd
cost_grad = grad(cost)

num_epochs = 10000
learning_rate = 0.001

# Manually implement gradient descent
for i in range(num_epochs):
    W = W - learning_rate * cost_grad(W)

# Print the final learned parameters.
print(W)
print(cost(W))
# Plot the original training data again, together with the polynomial we fit
plt.figure()
plt.plot(x, t, 'r.')
plt.plot(x, np.dot(feature_matrix, W), 'b-')