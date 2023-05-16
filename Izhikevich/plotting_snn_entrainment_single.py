import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import h5py


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.squeeze() / np.max(x.flatten())
    y = y.squeeze() / np.max(y.flatten())
    return float(np.mean((x - y)**2))


# load data set
fname = "results/entrainment/snn_entrainment_25.h5"
data = h5py.File(fname, "r")

res_dict = {"alpha": [], "dim": [], "seq": [], "train_loss_1": [], "train_loss_2": [], "test_loss_1": [],
            "test_loss_2": [], "kernel_width": [], "kernel_variance": []}

for i in range(len(data)-1):

    g = data[f"{i}"]
    res_dict["alpha"].append(np.asarray(g["alpha"]))
    res_dict["dim"].append(np.asarray(g["dimensionality"]))
    res_dict["seq"].append(np.asarray(g["sequentiality"]))
    res_dict["train_loss_1"].append(mse(np.asarray(g["targets"][0]), np.asarray(g["train_predictions"][0])))
    res_dict["train_loss_2"].append(mse(np.asarray(g["targets"][1]), np.asarray(g["train_predictions"][1])))
    test_losses = [mse(np.asarray(g["targets"][0]), np.asarray(sig)) for sig in g["test_predictions"][0]]
    res_dict["test_loss_1"].append(np.mean(test_losses))
    test_losses = [mse(np.asarray(g["targets"][1]), np.asarray(sig)) for sig in g["test_predictions"][1]]
    res_dict["test_loss_2"].append(np.mean(test_losses))

print(f"Delta: {float(np.asarray(data['sweep']['Delta']))}")
fig, axes = plt.subplots(ncols=4, figsize=(12, 4))
ax = axes[0]
ax.plot(res_dict["alpha"], res_dict["dim"])
ax.set_title("dim")
ax.set_xlabel("alpha")
ax = axes[1]
ax.plot(res_dict["alpha"], res_dict["seq"])
ax.set_title("seq")
ax.set_xlabel("alpha")
ax = axes[2]
ax.plot(res_dict["alpha"], res_dict["train_loss_1"])
ax.set_title("train loss")
ax.set_xlabel("alpha")
ax = axes[3]
ax.plot(res_dict["alpha"], res_dict["test_loss_1"])
ax.set_title("test loss")
ax.set_xlabel("alpha")
plt.tight_layout()
plt.show()
