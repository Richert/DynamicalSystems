import matplotlib.pyplot as plt
import pickle

# load data
###########

# load data from disk
file = "161001"
data_set = "trujilo_2019"
path = "/home/richard-gast/Documents/results/"
data = pickle.load(open(f"{path}/{data_set}/{file}_fitting_results.pkl", "rb"))

# extract relevant fields
target_fr = data["target_fr"]
fitted_fr = data["fitted_fr"]
target_psd = data["target_psd"]
fitted_psd = data["fitted_psd"]
freqs = data["freqs"]
results = data["fitting_results"]
loss = data["loss"]
organoid = 1 # data["organoid"]
keys = [] #data["param_keys"]

# print winner summary
print(f"Best fit for organoid {organoid} at {file} (loss = {loss})")
for key, val in zip(keys, results.x):
    print(f"{key} = {val}")

# plotting
##########

fig, axes = plt.subplots(nrows=2, figsize=(12, 8))
fig.suptitle(f"Fitting results for organoid {file}")

# time series
ax = axes[0]
ax.plot(target_fr, label="target")
ax.plot(fitted_fr, label="fit")
ax.set_xlabel("time (s)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("Mean-field dynamics")
ax.legend()

# PSD
ax = axes[1]
ax.plot(freqs, target_psd, label="target")
ax.plot(freqs, fitted_psd, label="fit")
ax.set_xlabel("frequency")
ax.set_ylabel("log(psd)")
ax.set_title("Power spectrum")
ax.legend()

plt.tight_layout()
plt.show()
