import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import h5py


# load data set
fname = "results/entrainment/snn_entrainment_0.h5"
data = h5py.File(fname, "r")

res_dict = {"alpha": [], "dim": [], "seq": [], "train_loss": [], "test_loss": [], "kernel_width": [],
            "kernel_variance": []}

for i in range(len(data)-1):

    g = data[f"{i}"]
    res_dict["alpha"].append(np.asarray(g["alpha"]))
    res_dict["dim"].append(np.asarray(g["dimensionality"]))
    res_dict["seq"].append(np.asarray(g["sequentiality"]))

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
ax = axes[0]
ax.plot(res_dict["alpha"], res_dict["dim"])
ax = axes[1]
ax.plot(res_dict["alpha"], res_dict["seq"])
plt.tight_layout()
plt.show()
