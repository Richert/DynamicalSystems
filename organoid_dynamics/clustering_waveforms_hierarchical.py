from custom_functions import *
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sb
import pickle

# read in data
path = "/home/richard-gast/Documents/results"
dataset = "trujilo_2019"
dist_data = pickle.load(open(f"{path}/{dataset}/{dataset}_waveform_distances.pkl", "rb"))
wave_data = pd.read_csv(f"{path}/{dataset}/{dataset}_waveforms.csv",
                   header=[0, 1, 2], index_col=0)
distances = ["dtw", "cc"]
n_clusters = 6

# loop over all distance types
for key in distances:

    # run hierarchical clustering on distance matrix
    D = dist_data[key]
    try:
        D_condensed = squareform(D)
    except ValueError:
        D_condensed = D
    Z = linkage(D_condensed, method="ward")
    clusters = cut_tree(Z, n_clusters=n_clusters)

    # plot prototypical waveforms
    proto_waves = get_cluster_prototypes(clusters.squeeze(), wave_data, reduction_method="random")
    fig, ax = plt.subplots(figsize=(12, 5))
    for sample, wave in proto_waves.items():
        ax.plot(wave / np.max(wave), label=sample)
    ax.set_ylabel("firing rate")
    ax.set_xlabel("time (ms)")
    ax.legend()
    ax.set_title(f"Normalized cluster waveforms for distance measure {key}")
    plt.tight_layout()

    # plot clustering results
    ax = sb.clustermap(D, row_linkage=Z, figsize=(12, 9))
    plt.title(f"Distance matrix and dendrogram for distance measure {key}")
    plt.tight_layout()

plt.show()
