from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import cdist_dtw
from tslearn.clustering import TimeSeriesKMeans
from custom_functions import *
import numpy as np
import pickle
import matplotlib.pyplot as plt

# read in data
path = "/home/richard"
dataset = "trujilo_2019"
data = pd.read_csv(f"{path}/data/{dataset}/{dataset}_waveforms.csv",
                   header=[0, 1, 2], index_col=0)

# clustering parameters
n_clusters = 5
max_iter = 100
tol = 1e-5
n_init = 3
metric = "euclidean"
max_iter_barycenter = 100
n_jobs = 15
plotting = False

# reduce data
age = 82
organoid = None
normalize = False
data_reduced = reduce_df(data, age=age, organoid=organoid)
# fig, ax = plt.subplots(figsize=(12, 4))
# data_reduced.plot(ax=ax)
# plt.show()
data_norm = data_reduced.values.T
if normalize:
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_norm.T).T

# run kmeans
km = TimeSeriesKMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol, n_init=n_init, metric=metric,
                      max_iter_barycenter=max_iter_barycenter, n_jobs=n_jobs, verbose=1)
km.fit(data_norm)

# get cluster prototypes and labels
cluster_labels = km.labels_
proto_waves = np.asarray(km.cluster_centers_).squeeze()

# calculate distance between waves
calc_dists = False
if calc_dists:
    D = cdist_dtw(data_norm[:, :], n_jobs=n_jobs)
    np.save(f"{path}/results/{dataset}/{n_clusters}cluster_waveform_distances.npy", D)
else:
    D = np.load(f"{path}/results/{dataset}/{n_clusters}cluster_waveform_distances.npy")

# get cluster waveforms
waveforms = get_cluster_prototypes(cluster_labels, data_reduced, reduction_method=None)

# save results
results = {"D": D, "labels": cluster_labels, "cluster_centroids": proto_waves, "waveforms": waveforms}
pickle.dump(results, open(f"{path}/results/{dataset}/{n_clusters}cluster_kmeans_results.pkl", "wb"))

if plotting:

    # plot prototypical waveforms
    fig, ax = plt.subplots(figsize=(12, 5))
    for sample, wave in enumerate(proto_waves):
        ax.plot(wave, label=sample)
    ax.set_ylabel("firing rate")
    ax.set_xlabel("time (ms)")
    ax.legend()
    ax.set_title(f"Normalized cluster waveforms")
    plt.tight_layout()

    # plot clustering results
    step = 15
    idx = np.argsort(cluster_labels)
    D = D[idx, :]
    D = D[:, idx]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(D, aspect="auto", interpolation="none", cmap="cividis")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(0, len(cluster_labels), step), labels=idx[::step])
    ax.set_yticks(np.arange(0, len(cluster_labels), step), labels=idx[::step])
    ax.set_ylabel("waveforms")
    ax.set_xlabel("waveforms")
    plt.title(f"Distance matrix")
    plt.tight_layout()
    plt.show()
