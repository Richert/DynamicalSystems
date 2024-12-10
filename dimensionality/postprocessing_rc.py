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
start = 20

# load data
results = {"rep": [], "g": [], "Delta": [], iv: [],
           "s_mean": [], "s_std": [], "s_norm": [], "ff_mean": [], "ff_max": [],
           "tau_ir": [], "offset_ir": [], "amp_ir": [],
           "patrec_loss": [], "patrec_tau": [], "K_diag": [], "K_magnitude": [], "funcgen_loss": [],
           "dim_ss": [], "dim_ss_r": [], "dim_ss_c": [], "dim_ss_rc": [],
           "dim_ir": [], "dim_ir_r": [], "dim_ir_c": [], "dim_ir_rc": [],
           "dim_sep": [], "dim_sep_r": [], "dim_sep_c": [], "dim_sep_rc": [],
           }
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
            results["dim_ss_r"].append(data["dim_ss_r"])
            results["dim_ss_c"].append(data["dim_ss_c"])
            results["dim_ss_rc"].append(data["dim_ss_rc"])
            results["s_mean"].append(np.mean(data["s_mean"])*1e3)
            results["s_std"].append(np.mean(data["s_std"]))
            results["s_norm"].append(results["s_std"][-1]*1e3/results["s_mean"][-1])
            results["ff_mean"].append(np.nanmean(data["ff_within"][-1]))
            results["ff_max"].append(np.nanmax(data["ff_within"][-1]))

            # impulse response analysis
            results["dim_ir"].append(data["dim_ir"])
            results["dim_ir_r"].append(data["dim_ir_r"])
            results["dim_ir_c"].append(data["dim_ir_c"])
            results["dim_ir_rc"].append(data["dim_ir_rc"])
            results["tau_ir"].append(data["params_ir"][-2])
            results["offset_ir"].append(data["params_ir"][0])
            results["amp_ir"].append(data["params_ir"][2])

            # separability analysis
            results["dim_sep"].append(data["dim_sep"])
            results["dim_sep_r"].append(data["dim_sep_r"])
            results["dim_sep_c"].append(data["dim_sep_c"])
            results["dim_sep_rc"].append(data["dim_sep_rc"])

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
