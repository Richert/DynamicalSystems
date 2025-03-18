import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import cdist_dtw
from tslearn.clustering import TimeSeriesKMeans
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from custom_functions import *


# read in data
path = "/home/richard"
dataset = "trujilo_2019"
data = pd.read_csv(f"{path}/data/{dataset}/{dataset}_waveforms.csv",
                   header=[0, 1, 2], index_col=0)

# reduce data
age = None
organoid = None
normalize = "True"
data_reduced = reduce_df(data, age=age, organoid=organoid)
# fig, ax = plt.subplots(figsize=(12, 4))
# data_reduced.plot(ax=ax)
# plt.show()
data_norm = data_reduced.values.T
if normalize:
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_norm)

# run kmeans
n_jobs = 80
km = TimeSeriesKMeans(n_clusters=6, max_iter=100, tol=1e-6, n_init=3, metric="dtw", max_iter_barycenter=100,
                      n_jobs=n_jobs, verbose=1)
km.fit(data_norm)

# calculate distance between waves
calc_dists = True
if calc_dists:
    D = cdist_dtw(data_norm[:, :], n_jobs=n_jobs)
    np.save(f"{path}/results/{dataset}/{dataset}_waveform_distances", D)
else:
    D = np.load(f"{path}/results/{dataset}/{dataset}_waveform_distances.npy")

# save prototypical waveforms
proto_waves = np.asarray(km.cluster_centers_).squeeze()
np.save(f"{path}/results/{dataset}/{dataset}_cluster_centers_kmeans.npy", proto_waves)

# # plot prototypical waveforms
# fig, ax = plt.subplots(figsize=(12, 5))
# for sample in range(proto_waves.shape[0]):
#     ax.plot(proto_waves[sample], label=sample)
# ax.set_ylabel("firing rate")
# ax.set_xlabel("time (ms)")
# ax.legend()
# ax.set_title("Prototypical waveforms fur clusters")
# plt.tight_layout()
#
# # plot clustering results
# fig, ax = plt.subplots(figsize=(10, 8))
# D = D[km.labels_, :]
# D = D[:, km.labels_]
# im = ax.imshow(D, interpolation="none", aspect="equal", cmap="cividis")
# plt.colorbar(im, ax=ax)
# plt.show()
