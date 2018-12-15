
# coding: utf-8

# In[ ]:

import scipy.io
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from time import time


# In[ ]:

def sharpNuc(Z, kappa):
    #SHARPNUC This function implements the sharp operator for the nuclear norm ball constraint. .
    
    # Implement sharp operator here!
    
    return 


# In[ ]:

data = scipy.io.loadmat('./dataset/ml-100k/ub_base')  # load 100k dataset

Rating = data['Rating'].flatten()
UserID = data['UserID'].flatten() - 1  # Python indexing starts from 0 whereas Matlab from 1
MovID = data['MovID'].flatten() - 1    # Python indexing starts from 0 whereas Matlab from 1

nM = np.amax(data['MovID'])
nU = np.amax(data['UserID'])

Z = csr_matrix((Rating, (MovID, UserID)),shape=(nM, nU),dtype=float)
kappa = 5000

tstart = time()
Z_proj = sharpNuc(Z, kappa)
elapsed = time() - tstart
print "sharp of 100k data takes ",elapsed," sec"


# In[ ]:

# NOTE: This one can take few minutes!
data = scipy.io.loadmat('./dataset/ml-1m/ml1m_base')  # load 1M dataset

Rating = data['Rating'].flatten()
UserID = data['UserID'].flatten() - 1  # Python indexing starts from 0 whereas Matlab from 1
MovID = data['MovID'].flatten() - 1    # Python indexing starts from 0 whereas Matlab from 1

nM = np.amax(data['MovID'])
nU = np.amax(data['UserID'])

Z = csr_matrix((Rating, (MovID, UserID)),shape=(nM, nU),dtype=float)
kappa = 5000

tstart = time()
Z_proj = sharpNuc(Z, kappa)
elapsed = time() - tstart
print "sharp of 1M data takes ",elapsed," sec"

