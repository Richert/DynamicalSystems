import pickle
import os
import numpy as np
from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d

# condition
iv = "ei_ratio"
condition = "rc_eir"
path = "/media/richard/results/dimensionality"

# loss meta parameters
sigma = 5
threshold = 0.1
start = 15

# load data
results = {"rep": [], "g": [], "Delta": [], iv: [], "dim_ss": [], "s_mean": [], "s_std": [], "s_norm": [],
           "dim_ir": [], "tau_ir": [], "offset_ir": [], "amp_ir": [],
           "patrec_loss": [], "patrec_tau": [], "K_diag": [], "K_magnitude": [], "funcgen_loss": [],
           "dim_ir_reduced": [], "dim_ss_reduced": [], "dim_ir_centered": [], "dim_ss_centered": []}
for file in os.listdir(path):
    if file[:len(condition)] == condition:

        if "summary" not in file:

            # load data
            data = pickle.load(open(f"{path}/{file}", "rb"))

            # condition information
            f = file.split("_")
            rep = int(f[-1].split(".")[0])
            results["rep"].append(rep)
            results["g"].append(data["g"])
            results["Delta"].append(data["Delta"])
            results[iv].append(data[iv])

            # steady-state analysis
            results["dim_ss"].append(data["dim_ss"])
            results["dim_ss_centered"].append(data["dim_ss_centered"])
            results["dim_ss_reduced"].append(data["dim_ss_reduced"])
            results["s_mean"].append(np.mean(data["s_mean"])*1e3)
            results["s_std"].append(np.mean(data["s_std"]))
            results["s_norm"].append(results["s_std"][-1]*1e3/results["s_mean"][-1])

            # impulse response analysis
            results["dim_ir"].append(data["dim_ir"])
            results["dim_ir_centered"].append(data["dim_ir_centered"])
            results["dim_ir_reduced"].append(data["dim_ir_reduced"])
            results["tau_ir"].append(data["params_ir"][-2])
            results["offset_ir"].append(data["params_ir"][0])
            results["amp_ir"].append(data["params_ir"][2])

            # pattern recognition task
            predictions = data["patrec_predictions"]
            targets = data["patrec_targets"]
            loss = np.asarray([np.mean((t-p)**2) for t, p in zip(predictions, targets)])[start:]
            loss_smoothed = np.asarray(gaussian_filter1d(loss, sigma=sigma))
            results["patrec_loss"].append(np.nanmean(loss_smoothed))
            try:
                idx = np.argwhere(loss_smoothed > threshold).squeeze()[0]
            except IndexError:
                idx = 0
            results["patrec_tau"].append(idx*2.0)

            # function generation task
            predictions = data["funcgen_predictions"]
            targets = data["funcgen_targets"]
            loss = np.asarray([np.mean((t - p) ** 2) for t, p in zip(predictions, targets)])[start:]
            loss_smoothed = np.asarray(gaussian_filter1d(loss, sigma=sigma))
            results["funcgen_loss"].append(np.nanmean(loss_smoothed))

            # kernel statistics
            K_diag = data["K_diag"]
            results["K_diag"].append(np.var(K_diag))
            K_mean, K_var = data["K_mean"], data["K_var"]
            results["K_magnitude"].append(np.mean(K_mean/(K_var + 1e-9)))

# create dataframe
df = DataFrame.from_dict(results)
df.to_pickle(f"{path}/{condition}_summary.pkl")
