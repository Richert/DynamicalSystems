import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(8)
import pickle
import numpy as np
from pyrecu import modularity, sort_via_modules
import matplotlib.pyplot as plt
import sys
cond = 1 #sys.argv[-1]


# preparations
##############

# load rnn data
data = pickle.load(open(f"results/rnn_{cond}.p", "rb"))
data['modules'] = []
data['adjacency'] = []
results = data['results']
etas = data['etas']

# analysis
##########

for res in results:

    # calculate modularity
    modules, A, nodes = modularity(res['s'].T, threshold=0.1, min_connections=5, min_nodes=5, decorator=nb.njit,
                                   cross_corr_method='fft', parallel=True, fastmath=True)

    # re-arrange adjacency matrix according to modules
    C = sort_via_modules(A, modules)

    # store results
    data['adjacency'] = A
    data['modules'] = modules

    # visualize modules
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].imshow(A)
    ax[1].imshow(C, cmap='nipy_spectral')
    plt.show()
