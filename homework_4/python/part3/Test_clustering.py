import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh, svds, eigs
from math import sqrt
import scipy.io as sio
import random
import numpy.matlib

# fix the seed
random.seed( 3 )

# Load data
Problem = sio.loadmat('clustering_data1.mat')
C = np.double(Problem['C']) # euclidean distance matrix
N = np.int(Problem['N']) # number of data points
k = Problem['k'] # number of clusters
opt_val = Problem['opt_val'] # optimum value 


## Define operators
# We provide 4 operators:
# 1- A1: Linear operator that takes the row sums
# 2- At2: Conjugate of operator A1
# 3- A2: Linear operator that takes the column sums 
# 4- At2: Conjugate of operator A2
A1 = lambda x: np.sum(x, axis = 1)
At1 = lambda y: np.transpose(np.matlib.repmat(y, N, 1))
A2 = lambda x: np.sum(x, axis = 0)
At2 = lambda y: (np.matlib.repmat(y, N, 1))

b = np.double(np.ones(N))

# Plotting function
def plot_func(cur_iter, feasibility1,feasibility2, objective, X):
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.loglog(cur_iter, feasibility1)#, 'go--', linewidth=2, markersize=12))
    plt.xlabel('iteration',fontsize=15)
    plt.ylabel('$\|X1-1\|/1$',fontsize=15)
    plt.grid(True)

    plt.subplot(122)
    plt.loglog(cur_iter, feasibility2)
    plt.xlabel('iteration',fontsize=15)
    plt.ylabel('dist$(X, \mathbb{R}^{n}_+)$',fontsize=15)
    plt.grid(True)
    plt.show()

    #plt.subplot(223)
    obj_res = np.reshape(np.abs(objective - opt_val)/opt_val, (len(objective),))
    plt.figure(figsize=(12, 8))
    plt.loglog((cur_iter), (obj_res))
    plt.xlabel('iteration',fontsize=15)
    plt.ylabel('$(f(X) - f^*)/f^*$',fontsize=15)
    plt.title('Relative objective residual',fontsize=15)
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(X)
    plt.title('SDP solution',fontsize=15)
    plt.show()
    
    
    
def HomotopyCGM(kappa=10, maxit=np.int(1e3), beta0=1):
    # Initialize
    X = np.zeros((N,N))
    AX1_b = 0.0
    
    feasibility1 = [] # norm(A1(X)-b1)/norm(b1)
    feasibility2 = [] # dist(X, \mathcal{K})
    objective    = [] # f(x)
    cur_iter    = [] 
    
    #u = np.zeros((N,1))
    iter_track = np.unique(np.ceil(np.power(2, np.linspace(0,20,100))))    
    
    for iteration in range(1, maxit+1):
        
        # Update Step Size
        gamma = ???
        
        # Update beta
        beta_ = ???
        
        # Write down the vk to use in the lmo (eigenvalue routine)
        vk = ???
        vk = 0.5*(vk + vk.T)
        
        # Linear minimization oracle
        # use eigsh function with proper settings to calculate the lmo
        # See Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh 
        q, u = eigsh(???)
        u = sqrt(kappa)*u
        X_bar = np.outer(u,u)
        
        # Obtain A*Xbar - b
        AX_bar_b = A1(X_bar)-b
        
        # Update A*X - b
        AX1_b = (1.0-gamma)*AX1_b + gamma*(AX_bar_b)
        
        # Update X
        X = ???
        
        if any(iteration == iter_track) or iteration == maxit:
            #print(X)
            feasibility1.append(np.linalg.norm(AX1_b)/N)
            feasibility2.append(np.linalg.norm(np.minimum(X,0), ord='fro')) # distance to positive orthant
            objective.append(np.sum(C.flatten()*X.flatten()))
            cur_iter.append(iteration)
            print('{:03d} | {:.4e}| {:.4e}| {:.4e}|'.format(iteration, feasibility1[-1], feasibility2[-1],objective[-1]))
            
    return X, feasibility1, feasibility2, objective, cur_iter
    
# run the algorithm
X, feasibility1, feasibility2, objective, cur_iter = HomotopyCGM(10, np.int(1e3), 1)

# plot the results
plot_func(cur_iter, feasibility1,feasibility1, objective, X)