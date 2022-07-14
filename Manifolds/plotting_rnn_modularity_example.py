import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(8)
import pickle
import numpy as np
from pyrecu import modularity, sort_via_modules
import matplotlib.pyplot as plt

# load rnn data
cond = 5
data = pickle.load(open(f"results/rnn_simulations/rnn_{cond}.p", "rb"))
res = data['results']
in_vars = np.asarray(data['in_var'])
p = data['p']
W = data['W']

# extract target signals
in_var = 0.5
n = np.argmin(np.abs(in_vars-in_var))
signal = res[n]['v']

# z-transform membrane potentials
z = np.zeros_like(signal)
for idx in range(z.shape[1]):
    z_tmp = signal[:, idx]
    z_tmp -= np.mean(z_tmp)
    z_tmp /= np.max(np.abs(z_tmp))
    z[:, idx] = z_tmp

modules, A, nodes = modularity(z.T, threshold=0.1, min_connections=4, min_nodes=4, cross_corr_method='fft',
                               decorator=None)

# re-arrange adjacency matrix according to modules
C = sort_via_modules(A, modules)

# plotting
fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
W_tmp = W[nodes, :]
W_tmp = W_tmp[:, nodes]
ax[0].imshow(W_tmp)
ax[0].set_title('Synapses')
ax[1].imshow(A)
ax[1].set_title('Adjacency')
ax[2].imshow(C, cmap='nipy_spectral')
ax[2].set_title('Modules')
plt.show()
