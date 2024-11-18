import pickle
import os
import numpy as np
from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d

# condition
iv = "p"
condition = "rc_exc"
path = "/media/richard/results/dimensionality"

# loss meta parameters
sigma = 5
threshold = 0.05
start = 12

# load data
results = {"rep": [], "g": [], "Delta": [], iv: [], "dim_ss": [], "s_mean": [], "s_std": [], "s_norm": [],
           "dim_ir": [], "tau_ir": [], "offset_ir": [], "amp_ir": [], "min_loss": [], "max_loss": [], "tau_rc": []}
for file in os.listdir(path):
    if file[:len(condition)] == condition:

        # load data
        data = pickle.load(open(f"{path}/{file}", "rb"))

        # condition information
        f = file.split("_")
        if "N" in data:
            rep = int(f[-1].split(".")[0])
            results["rep"].append(rep)
            results["g"].append(data["g"])
            results["Delta"].append(data["Delta"])
            # results[iv].append(data[iv])
            results[iv].append(float(f[-2][1:]))

            # steady-state analysis
            results["dim_ss"].append(data["dim_ss"])
            results["s_mean"].append(np.mean(data["s_mean"])*1e3)
            results["s_std"].append(np.mean(data["s_std"]))
            results["s_norm"].append(results["s_std"][-1]*1e3/results["s_mean"][-1])

            # impulse response analysis
            results["dim_ir"].append(data["dim_ir"])
            results["tau_ir"].append(data["params_ir"][-2])
            results["offset_ir"].append(data["params_ir"][0])
            results["amp_ir"].append(data["params_ir"][2])

            # reservoir computing analysis
            predictions = data["predictions"]
            targets = data["targets"]
            loss = np.asarray([np.mean((t-p)**2) for t, p in zip(predictions, targets)])[start:]
            loss_smoothed = np.asarray(gaussian_filter1d(loss, sigma=sigma))
            results["min_loss"].append(np.min(loss_smoothed))
            results["max_loss"].append(np.max(loss_smoothed))
            try:
                idx = np.argwhere(loss_smoothed > threshold).squeeze()[0]
            except IndexError:
                idx = 0
            results["tau_rc"].append(idx*2.0)

# create dataframe
df = DataFrame.from_dict(results)
df.to_pickle(f"{path}/{condition}_summary.pkl")
