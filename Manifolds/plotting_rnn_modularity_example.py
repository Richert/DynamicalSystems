import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(8)
import pickle
import numpy as np
from pyrecu import modularity, sort_via_modules
import matplotlib.pyplot as plt

# load rnn data
cond = 0
data = pickle.load(open(f"results/rnn_simulations/rnn_{cond}.p", "rb"))
res = data['results']
etas = np.asarray(data['etas'])
p = data['p']
W = data['W']

# calculate modularity
eta = 50.0
n = np.argmin(np.abs(etas-eta))
modules, A, nodes = modularity(res[n]['s'].T, threshold=0.2, min_connections=5, min_nodes=5, cross_corr_method='fft',
                               parallel=True, fastmath=True, decorator=nb.njit)

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
