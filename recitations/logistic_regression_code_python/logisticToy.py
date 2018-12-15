import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from GD import GD 

data = scipy.io.loadmat('synthetic-data-noisy.mat')

A = np.concatenate((np.ones([data['features'].shape[0],1]), data['features']), axis=1) # for bias
b = data['labels']
# An example of python lambda function is below.
# This function computes the sigmoid of each entry of its input x.
sigmoid = lambda x: (1./(1.+ np.exp(-x)))
# NOTE: You can use the sigmoid above as a building block in fsx and gradfx
# i.e. expressions below may contain sigmoid(x).
# fx computes the objective (l-2 regularized) of input x
fx     = lambda x: (-np.sum(np.log(sigmoid(b * (A.dot(x)))), axis=0))
# gradf computes the gradient (l-2 regularized) of input x
gradf  = lambda x: (-A.T.dot((1. - sigmoid(b*(A.dot(x)))) * b))

# train the model
maxit             = ...?
stepsize          = ...?
x0                = ...?

xGD, obj          = GD(fx, gradf, stepsize, x0, maxit)

# 2d scatter plot
idx1 = np.where(data['labels'] == -1)[0]
idx2 = np.where(data['labels'] == 1)[0]
separating_line     = lambda x, ylim: (-x[0] - ylim*x[2])/x[1]

plt.figure(1)
plt.plot(data['features'][idx1, 0], data['features'][idx1, 1], 'rs')
plt.plot(data['features'][idx2, 0], data['features'][idx2, 1], 'bo')
plt.xlabel('x_1')
plt.ylabel('x_2')

plt.plot(separating_line(xGD, plt.ylim()), plt.ylim(), 'b-');
#plt.show()

# plot the convergence
plt.figure(2)
plt.loglog(obj - data['optval'])
plt.ylabel('f(x)-f^*')
plt.xlabel('iterations')
axes = plt.gca()
axes.set_ylim([1e-4,1e4])

plt.show()