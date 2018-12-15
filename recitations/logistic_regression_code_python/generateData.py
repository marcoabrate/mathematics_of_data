import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

p = 2
n = 100

x_true = np.random.randn(p, 1)
mu_true = np.random.randn(1,1)

features = np.random.randn(n,p)
labels = np.sign(features.dot(x_true) + mu_true)

idx1 = np.where(labels == -1)[0]
idx2 = np.where(labels == 1)[0]

plt.figure(1)
plt.plot(features[idx1, 0], features[idx1, 1], 'rs'
plt.plot(features[idx2, 0], features[idx2, 1], 'bo')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()