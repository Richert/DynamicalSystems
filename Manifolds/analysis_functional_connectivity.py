import pickle
import matplotlib.pyplot as plt
from pyrecu import cross_corr, modularity

# load data
data = pickle.load(open("results/snn_autonomous.pkl", "rb"))

# pre-compile cross-correlation function
cc_func = cross_corr
# cc_func(2, signals=data["s"].loc[:1000, :].values.T)

# calculate cross-correlation between time series
cc = cc_func(len(data["etas"]), signals=data["s"].loc[100000:, :].values.T, method="fft")

# saving
data["cc"] = cc
pickle.dump(data, open("results/snn_autonomous.pkl", "wb"))

# plotting
_, ax = plt.subplots()
im = ax.imshow(cc, aspect=1.0, interpolation="none")
plt.colorbar(mappable=im)
ax.set_title("CC")

plt.show()
