from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import cdist_dtw, cdist_soft_dtw, cdist_gak
from scipy.spatial.distance import cdist
from custom_functions import *
import pickle

# read in data
path = "/home/richard"
dataset = "trujilo_2019"
data = pd.read_csv(f"{path}/data/{dataset}/{dataset}_waveforms.csv",
                   header=[0, 1, 2], index_col=0)

# distances calculation settings
n_jobs = 80
dist_measures = {"dtw": cdist_dtw, "soft_dtw": cdist_soft_dtw, "cc": cdist}
dist_kwargs = {"dtw": {"n_jobs": n_jobs}, "soft_dtw": {}, "cc": {"metric": "correlation"}}

# reduce data
age = None
organoid = None
normalize = True
data_reduced = reduce_df(data, age=age, organoid=organoid)
data_norm = data_reduced.values.T
if normalize:
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_norm)

# calculate distance between waves
distances = {}
for key, func in dist_measures.items():
    D = func(data_norm[:, :], data_norm[:, :], **dist_kwargs[key])
    distances[key] = D

# save results
pickle.dump(distances, open(f"{path}/results/{dataset}/{dataset}_waveform_distances.pkl", "wb"))
