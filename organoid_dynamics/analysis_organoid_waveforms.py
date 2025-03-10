import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import cdist_dtw
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sb


def get_cluster_prototypes(clusters, waveforms: pd.DataFrame, method: str = "mean") -> dict:
    cluster_ids = np.unique(clusters)
    prototypes = {}
    for cluster_id in cluster_ids:
        ids = np.argwhere(clusters == cluster_id).squeeze()
        if method == "mean":
            prototypes[cluster_id] = waveforms.iloc[:, ids].mean(axis=1)
        elif method == "random":
            prototypes[cluster_id] = waveforms.iloc[:, np.random.choice(ids)]
        else:
            raise ValueError("Invalid prototype building choice")

    return prototypes

def reduce_df(df: pd.DataFrame, age: int = None, organoid: int = None) -> pd.DataFrame:

    if age is not None:
        return df[f"{age}"]
    if organoid is not None:
        return df.xs(f"{organoid}", level="well", axis="columns")
    return df


# read in data
path = "/home/richard-gast/Documents/data"
dataset = "trujilo_2019"
data = pd.read_csv(f"{path}/{dataset}/{dataset}_waveforms.csv",
                   header=[0, 1, 2], index_col=0)

# reduce data
age = None
organoid = None
normalize = "False"
data_reduced = reduce_df(data, age=age, organoid=organoid)
fig, ax = plt.subplots(figsize=(12, 4))
data_reduced.plot(ax=ax)
plt.show()
data_norm = data_reduced.values.T
if normalize:
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_norm)

# calculate distance between waves
D = cdist_dtw(data_norm[:, :], n_jobs=15)
np.save(f"{path}/{dataset}/waveform_distances", D)

# run hierarchical clustering on distance matrix
D_condensed = squareform(D)
Z = linkage(D_condensed, method="ward")
clusters = cut_tree(Z, n_clusters=6)

# plot prototypical waveforms
proto_waves = get_cluster_prototypes(clusters.squeeze(), data_reduced, method="random")
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
