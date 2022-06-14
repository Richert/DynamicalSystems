import numpy as np
from pyrecu import modularity
import pickle
import matplotlib.pyplot as plt

data = pickle.load(open("results/spn_rnn.p", "rb"))['results']

modules, A, nodes = modularity(data['s'].T, mode='same', threshold=30.0, min_connections=10, min_nodes=5)
C = np.zeros_like(A)
node_order = []
for module, (nodes_tmp, _) in modules.items():
    node_order.extend(list(nodes_tmp))
    idx = np.ix_(nodes_tmp, np.arange(0, A.shape[1]))
    idx2 = A[idx] > 0
    C_tmp = np.zeros(shape=idx2.shape, dtype=np.int32)
    C_tmp[idx2] = module
    C[idx] = C_tmp
C1 = np.zeros_like(C)
for i, n in enumerate(node_order):
    C1[i, :] = C[n, :]
C2 = np.zeros_like(C)
for i, n in enumerate(node_order):
    C2[:, i] = C1[:, n]

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(A)
ax[1].imshow(C2, cmap='nipy_spectral')
plt.show()
