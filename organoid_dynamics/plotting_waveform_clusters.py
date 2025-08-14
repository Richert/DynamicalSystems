import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from custom_functions import *
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform

# choose data to fit
dataset = "trujilo_2019"
n_clusters = 2

# define directories and file to fit
path = "/home/richard-gast/Documents"
save_dir = f"{path}/results/{dataset}"
load_dir = f"{path}/data/{dataset}"

# load data from file
data = pd.read_csv(f"{load_dir}/{dataset}_waveforms.csv", header=[0, 1, 2], index_col=0)
D = np.load(f"{load_dir}/{dataset}_waveform_distances.npy")

# reduce data
age = 82
organoid = None
normalize = True
data = reduce_df(data, age=age, organoid=organoid)

# run hierarchical clustering on distance matrix
D_condensed = squareform(D)
Z = linkage(D_condensed, method="ward")
clusters = cut_tree(Z, n_clusters=n_clusters)

# extract target waveform
proto_waves = get_cluster_prototypes(clusters.squeeze(), data, reduction_method="random")

# plot prototypical waveforms
fig, ax = plt.subplots(figsize=(12, 5))
for sample, wave in proto_waves.items():
    ax.plot(wave / np.max(wave), label=sample)
ax.set_ylabel("firing rate")
ax.set_xlabel("time (ms)")
ax.legend()
ax.set_title(f"Normalized cluster waveforms")
plt.tight_layout()

# plot clustering results
ax = sb.clustermap(D, row_linkage=Z, figsize=(12, 9))
plt.title(f"Distance matrix and dendrogram")
plt.tight_layout()
plt.show()
