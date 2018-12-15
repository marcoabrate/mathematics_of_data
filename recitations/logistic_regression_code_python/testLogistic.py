import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from GD import GD 

data = scipy.io.loadmat('breast-cancer.mat')

b = data['labels_train']
A = np.concatenate((np.ones([data['features_train'].shape[0],1]), data['features_train'].toarray()), axis=1) # for bias

# First order oracle
# An example of python lambda function is below.
# This function computes the sigmoid of each entry of its input x.
sigmoid = lambda x: (1./(1.+ np.exp(-x)))
# fx computes the objective (l-2 regularized) of input x
fx     = lambda x: (-np.sum(np.log(sigmoid(b * (A.dot(x)))), axis=0))
# gradf computes the gradient (l-2 regularized) of input x
gradf  = lambda x: (-A.T.dot((1. - sigmoid(b*(A.dot(x)))) * b))

# parameters
maxit             = ...?
stepsize          = ...?
x0                = ...?

# gradient descent
xGD, obj          = GD(fx, gradf, stepsize, x0, maxit)

# plot the convergence
plt.figure()
plt.loglog((obj - data['optval'])/data['optval'])
plt.ylabel('(f(x)-f^*)/f^*')
plt.xlabel('iterations')
axes = plt.gca()

plt.show()