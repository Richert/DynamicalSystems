import pickle
import os
import numpy as np
from pandas import DataFrame

# condition
iv = "spatial_dim"
condition = "eic"
path = "/media/richard/results/dimensionality"

# load data
results = {"rep": [], "g": [], "Delta": [], iv: [],
           "s_mean": [], "s_std": [], "s_norm": [], "ff_mean": [], "ff_max": [],
           "dim_ss": [], "dim_ss_c": [], "dim_ss_r": [], "dim_ss_rc": [],
           "dim_ir": [], "dim_ir_c": [], "dim_ir_r": [], "dim_ir_rc": [],
           "dim_sep": [], "dim_sep_c": [], "dim_sep_r": [], "dim_sep_rc": [],
           "tau_ir": [], "offset_ir": [], "amp_ir": [],
           "tau_k": [], "mean_k": [], "std_k": []}
for file in os.listdir(path):
    if file[:len(condition)] == condition:

        # load data
        data = pickle.load(open(f"{path}/{file}", "rb"))

        if "summary" not in file:

            # condition information
            f = file.split("_")
            rep = int(f[-1].split(".")[0])
            results["rep"].append(rep)
            results["g"].append(data["g"])
            results["Delta"].append(data["Delta"])
            results[iv].append(int(f[1][0]))

            # steady-state analysis
            results["s_mean"].append(np.mean(data["s_mean"])*1e3)
            results["s_std"].append(np.mean(data["s_std"]))
            results["s_norm"].append(results["s_std"][-1]*1e3/results["s_mean"][-1])
            results["ff_mean"].append(np.nanmean(data["ff_within"][-1]))
            results["ff_max"].append(np.nanmax(data["ff_within"][-1]))

            # dimensionality analysis
            for c1 in ["ss", "ir", "sep"]:
                for c2 in ["", "_c", "_r", "_rc"]:
                    results[f"dim_{c1 + c2}"].append(data[f"dim_{c1 + c2}"])

            # impulse response analysis
            results["tau_ir"].append(data["params_ir"][-2])
            results["offset_ir"].append(data["params_ir"][0])
            results["amp_ir"].append(data["params_ir"][2])

            # kernel analysis
            results["tau_k"].append(data["params_kernel"][-2])
            results["mean_k"].append(np.mean(data["K_diag"]))
            results["std_k"].append(np.std(data["K_diag"]))

# create dataframe
df = DataFrame.from_dict(results)
df.to_pickle(f"{path}/{condition}_summary.pkl")
