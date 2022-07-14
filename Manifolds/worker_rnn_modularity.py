import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(6)
import pickle
import numpy as np
from pyrecu import modularity
import sys
cond = 2#sys.argv[-1]


# preparations
##############

# load rnn data
results = pickle.load(open(f"results/rnn_simulations/rnn_{cond}.p", "rb"))['results']

# analysis
##########

data = dict()
data['modules'] = []
data['adjacency'] = []
data['nodes'] = []

for res in results:

    # z-transform membrane potentials
    z = np.zeros_like(res['v'])
    for idx in range(z.shape[1]):
        z_tmp = res['v'][:, idx]
        z_tmp -= np.mean(z_tmp)
        z_tmp /= np.max(np.abs(z_tmp))
        z[:, idx] = z_tmp

    # calculate modularity
    modules, A, nodes = modularity(z.T, threshold=0.1, min_connections=4, min_nodes=4, decorator=None,
                                   cross_corr_method='fft')

    # import matplotlib.pyplot as plt
    # from pyrecu import sort_via_modules
    # C = sort_via_modules(A, modules)
    #
    # fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
    # ax[0].imshow(A)
    # ax[0].set_title('Adjacency')
    # ax[1].imshow(C, cmap='nipy_spectral')
    # ax[1].set_title('Modules')
    # plt.show()

    # store results
    data['adjacency'].append(A)
    data['modules'].append(modules)
    data['nodes'].append(nodes)

pickle.dump(data, open(f"results/mod_{cond}.p", "wb"))
