import numpy as np
from pyrecu import modularity
import pickle
import matplotlib.pyplot as plt

data = pickle.load(open("results/spn_rnn.p", "rb"))['results']

modules, A, nodes = modularity(data['s'].T, mode='same', threshold=40.0, min_connections=4, min_nodes=4)
C = np.zeros_like(A)
for module, (nodes_tmp, _) in modules.items():
    idx = np.ix_(nodes_tmp, nodes_tmp)
    idx2 = A[idx] > 0
    C_tmp = np.zeros(shape=idx2.shape, dtype=np.int32)
    C_tmp[idx2] = module
    C[idx] = C_tmp

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(A)
ax[1].imshow(C)
plt.show()
