import matplotlib.pyplot as plt
import pickle
import numpy as np
from sbi import analysis as analysis

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
time = data["time"]
organoid = data["organoid"]
age = data["age"]
fitted_params = data["posterior_samples"]
param_keys = data["param_keys"] if "param_keys" in data else \
    ["C", "k", "Delta", "kappa", "tau_u", "g", "tau_s", "s_ext", "noise_lvl", "sigma"]

# print winner summary
print(f"Average posterior for organoid {organoid} at {file} (age = {age})")
for key, val in zip(param_keys, np.mean(fitted_params, axis=0)):
    print(f"{key} = {val}")

# plotting
##########

# resulting model dynamics
fig, axes = plt.subplots(nrows=2, figsize=(12, 8))
fig.suptitle(f"Fitting results for organoid {file} (age = {age})")

ax = axes[0]
ax.plot(time, target_fr, label="target")
ax.plot(time[:len(fitted_fr)], fitted_fr, label="fit")
ax.set_xlabel("time (s)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("Mean-field dynamics")
ax.legend()

ax = axes[1]
ax.plot(freqs, target_psd, label="target")
ax.plot(freqs[:len(fitted_psd)], fitted_psd, label="fit")
ax.set_xlabel("frequency")
ax.set_ylabel("log(psd)")
ax.set_title("Power spectrum")
ax.legend()

plt.tight_layout()

# parameter posterior distributions
fig, axes = analysis.pairplot(
    fitted_params,
    figsize=(12, 9),
    points=np.mean(fitted_params, axis=0),
    points_offdiag={"markersize": 6},
    points_colors="r",
    labels=param_keys,
)
plt.tight_layout()

plt.show()
