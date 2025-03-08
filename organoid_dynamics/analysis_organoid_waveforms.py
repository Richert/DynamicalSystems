import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import cdist_dtw
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sb


def get_cluster_prototypes(clusters, waveforms: np.ndarray, method: str = "mean") -> dict:
    cluster_ids = np.unique(clusters)
    prototypes = {}
    for cluster_id in cluster_ids:
        ids = np.argwhere(clusters == cluster_id).squeeze()
        if method == "mean":
            prototypes[cluster_id] = np.mean(waveforms[ids, :], axis=0)
        elif method == "random":
            prototypes[cluster_id] = waveforms[np.random.choice(ids), :]
        else:
            raise ValueError("Invalid prototype building choice")

    return prototypes


# read in data
data = pd.read_csv("/home/richard-gast/Documents/data/trujilo_2019/trujilo_2019_waveforms.csv",
                   header=[0, 1, 2], index_col=0)

# normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data["82"].values.T)

# calculate distance between waves
D = cdist_dtw(normalized_data, n_jobs=15)

# run hierarchical clustering on distance matrix
D_condensed = squareform(D)
Z = linkage(D_condensed, method="centroid")
clusters = cut_tree(Z, n_clusters=4)

# plot prototypical waveforms
proto_waves = get_cluster_prototypes(clusters.squeeze(), normalized_data, method="random")
fig, ax = plt.subplots(figsize=(12, 5))
for sample, wave in proto_waves.items():
    ax.plot(wave, label=sample)
ax.set_ylabel("firing rate")
ax.set_xlabel("time (ms)")
ax.legend()
ax.set_title("Prototypical waveforms fur clusters")
plt.tight_layout()

# plot clustering results
ax = sb.clustermap(D, row_linkage=Z, figsize=(12, 9))
plt.show()
