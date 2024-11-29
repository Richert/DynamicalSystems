import pickle
import os
import numpy as np
from pandas import DataFrame

# condition
iv = "Delta_i"
condition = "dim2_eic"
path = "/media/richard/results/dimensionality"

# load data
results = {"rep": [], "g": [], "Delta": [], iv: [], "dim_ss": [], "s_mean": [], "s_std": [], "s_norm": [],
           "dim_ir": [], "tau_ir": [], "offset_ir": [], "amp_ir": [], "dim_ir_nc": [], "dim_ss_nc": []}
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
            results["Delta"].append(data["Delta_e"])
            results[iv].append(data[iv])

            # steady-state analysis
            results["dim_ss"].append(data["dim_ss"])
            results["dim_ss_nc"].append(data["dim_ss_nc"])
            results["s_mean"].append(np.mean(data["s_mean"])*1e3)
            results["s_std"].append(np.mean(data["s_std"]))
            results["s_norm"].append(results["s_std"][-1]*1e3/results["s_mean"][-1])

            # impulse response analysis
            results["dim_ir"].append(data["dim_ir"])
            results["dim_ir_nc"].append(data["dim_ir_nc"])
            results["tau_ir"].append(data["params_ir"][-2])
            results["offset_ir"].append(data["params_ir"][0])
            results["amp_ir"].append(data["params_ir"][2])

# create dataframe
df = DataFrame.from_dict(results)
df.to_pickle(f"{path}/{condition}_summary.pkl")
