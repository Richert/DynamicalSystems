import pickle
import matplotlib.pyplot as plt
from pyrecu import cross_corr, modularity
from scipy.spatial.distance import pdist, squareform
import numpy as np

# load data
fname = "snn_data2"
data = pickle.load(open(f"results/{fname}.pkl", "rb"))

# calculate cross-correlation between time series
cutoff = 1000
var = "s"
cc = cross_corr(len(data["etas"]), signals=data[var].iloc[cutoff:, :].v1.T, method="fft")
#cc = squareform(pdist(data[var].iloc[cutoff:, :].values.T, metric="euclidean"))

# saving
data["cc"] = cc
pickle.dump(data, open(f"results/{fname}.pkl", "wb"))

# plotting
_, ax = plt.subplots()
im = ax.imshow(cc, aspect=1.0, interpolation="none")
plt.colorbar(mappable=im)
ax.set_title("CC")

plt.show()
