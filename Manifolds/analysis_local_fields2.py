import numpy as np
import matplotlib.pyplot as plt
import pickle
from pyrecu import modularity, sort_via_modules
from scipy.spatial.distance import pdist, squareform
from rectipy import normalize

plt.rc('text', usetex=True)

# load data
###########

fname = "snn_data5"
data = pickle.load(open(f"results/{fname}.pkl", "rb"))
etas = data["etas"]
J = data["J"]
CC = data["cc"]
N = len(etas)

# local field analysis
######################

# calculate local fields
fields = [etas[np.argwhere(J[i, :] > 0).squeeze()] for i in range(N)]

# calculate local field pdfs
eta_grid = np.linspace(np.min(etas), np.max(etas), num=50)
field_rhos = []
for i in range(N):
    grid_fields = np.zeros_like(eta_grid) + 1e-10
    for eta in fields[i]:
        idx = np.argmin(np.abs(eta_grid - eta))
        grid_fields[idx] += 1
    field_rhos.append(grid_fields/len(fields[i]))

# calculate the distance between local fields
field_dist = squareform(pdist(np.asarray(field_rhos), metric="correlation"))

# calculate modularity of local fields
X = normalize(field_dist, mode="minmax", row_wise=False) - 1
X[np.eye(N, dtype=bool)] = 0.0
modules_field, A_field, nodes_field = modularity(X, threshold=0.1, min_nodes=5, min_connections=5)

# calculate the modularity of the functional connectivities
modules_fc, A_fc, nodes_fc = modularity(CC, threshold=0.1, min_connections=5, min_nodes=5)

# visualization
###############

print(f"Number of modules detected in local fields: {len(modules_field)}")
print(f"Number of modules detected in cross-correlations: {len(modules_fc)}")

# local field distance
_, ax = plt.subplots()
im = ax.imshow(field_dist, aspect=1.0, interpolation="none")
plt.colorbar(mappable=im)
ax.set_title(r"Local Field Distances")

# local field adjacency
_, ax = plt.subplots()
im = ax.imshow(A_field, aspect=1.0, interpolation="none")
plt.colorbar(mappable=im)
ax.set_title(r"Local Field Adjacency")

# functional connections
_, ax = plt.subplots()
im = ax.imshow(A_fc, aspect=1.0, interpolation="none")
plt.colorbar(mappable=im)
ax.set_title(r"Functional Adjacency")

plt.show()
k = 200

# local field adjacency
_, ax = plt.subplots()
im = ax.imshow(sort_via_modules(A_field, modules_field), aspect=1.0, interpolation="none")
ax.set_xlabel(nodes_field[::k])
ax.set_ylabel(nodes_field[::k])
plt.colorbar(mappable=im)
ax.set_title(r"Local Field Adjacency")

# functional connections
_, ax = plt.subplots()
im = ax.imshow(sort_via_modules(A_fc, modules_fc), aspect=1.0, interpolation="none")
ax.set_xlabel(nodes_fc[::k])
ax.set_ylabel(nodes_fc[::k])
plt.colorbar(mappable=im)
ax.set_title(r"Functional Adjacency")

plt.show()
