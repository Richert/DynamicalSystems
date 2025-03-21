from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import cdist_dtw
from tslearn.clustering import TimeSeriesKMeans
from custom_functions import *
import numpy as np
import pickle

# read in data
path = "/home/richard"
dataset = "trujilo_2019"
data = pd.read_csv(f"{path}/data/{dataset}/{dataset}_waveforms.csv",
                   header=[0, 1, 2], index_col=0)

# clustering parameters
n_clusters = 9
max_iter = 100
tol = 1e-5
n_init = 3
metric = "dtw"
max_iter_barycenter = 100
n_jobs = 80

# reduce data
age = None
organoid = None
normalize = True
data_reduced = reduce_df(data, age=age, organoid=organoid)
# fig, ax = plt.subplots(figsize=(12, 4))
# data_reduced.plot(ax=ax)
# plt.show()
data_norm = data_reduced.values.T
if normalize:
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_norm)

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
    np.save(f"{path}/results/{dataset}/{dataset}_waveform_distances", D)
else:
    D = np.load(f"{path}/results/{dataset}/{dataset}_waveform_distances.npy")

# get cluster waveforms
waveforms = get_cluster_prototypes(cluster_labels, data, reduction_method=None)

# save results
results = {"D": D, "labels": cluster_labels, "cluster_centroids": proto_waves, "waveforms": waveforms}
pickle.dump(results, open(f"{path}/results/{dataset}/{dataset}_kmeans_results.npy", "wb"))
