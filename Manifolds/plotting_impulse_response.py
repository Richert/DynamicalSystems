import sys
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt

# settings
##########

print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6
cmap = plt.get_cmap('plasma')

# prepare data
##############

# parameters of interest
p1 = "alpha"
p2 = "Delta"

# collect data of interest in dataframe
path = "results/ir_delta"
fn = "ir_delta"
cols = ["trial", "train_score", "test_score", "dimensionality", "K_diff", "alpha", "p"]
data = pd.DataFrame(columns=cols)
for f in os.listdir(path):
    if fn in f:
        file = pickle.load(open(f"{path}/{f}", "rb"))
        data_tmp = pd.DataFrame(columns=cols, data=np.zeros((1, len(cols))))
        for idx in range(len(file["s"])):
            data_tmp.loc[0, "trial"] = idx
            data_tmp.loc[0, "train_score"] = np.mean(file["train_scores"].values[idx, :])
            data_tmp.loc[0, "test_score"] = np.mean(file["test_scores"].values[idx, :])
            data_tmp.loc[0, "dimensionality"] = file["X_dim"][idx]
            #data_tmp.loc[0, "K_diff"] = file["K_diff"][idx]
            data_tmp.loc[0, p1] = file["sweep"][p1]
            data_tmp.loc[0, p2] = file["sweep"][p2]
            data = pd.concat([data, data_tmp], axis=0)

# prepare data for plotting
p1_vals = np.unique(data[p1].values)
p2_vals = np.unique(data[p2].values)
n = len(p1_vals)
m = len(p2_vals)
test_score_mean = np.zeros((n, m))
train_score_mean = np.zeros_like(test_score_mean)
dim_mean = np.zeros_like(test_score_mean)
#kernel_diff_mean = np.zeros_like(test_score_mean)
for row in range(n):
    for col in range(m):
        v1, v2 = p1_vals[row], p2_vals[col]
        idx1 = data.loc[:, p1] == v1
        idx2 = data.loc[:, p2] == v2
        idx = np.argwhere(idx1.values * idx2.values == 1.0).squeeze()
        points = data.iloc[idx, :]
        test_score_mean[row, col] = np.mean(points["test_score"].values)
        train_score_mean[row, col] = np.mean(points["train_score"].values)
        dim_mean[row, col] = np.mean(points["dimensionality"].values)
        #kernel_diff_mean[row, col] = np.mean(points["K_diff"].values)

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# test scores
ax = fig.add_subplot(grid[0, 0])
im = ax.imshow(test_score_mean[:, :], aspect=0.8)
ax.set_ylabel(p1)
ax.set_yticks(np.arange(len(p1_vals)), labels=p1_vals)
ax.set_xlabel(p2)
ax.set_xticks(np.arange(len(p2_vals)), labels=p2_vals)
ax.set_title("Test Scores")
fig.colorbar(im, ax=ax, shrink=0.4)

# training scores
ax = fig.add_subplot(grid[0, 1])
im = ax.imshow(train_score_mean[:, :], aspect=0.8)
ax.set_ylabel(p1)
ax.set_yticks(np.arange(len(p1_vals)), labels=p1_vals)
ax.set_xlabel(p2)
ax.set_xticks(np.arange(len(p2_vals)), labels=p2_vals)
ax.set_title("Training Scores")
fig.colorbar(im, ax=ax, shrink=0.4)

# dimensionality
ax = fig.add_subplot(grid[1, 0])
im = ax.imshow(dim_mean[:, :], aspect=0.8)
ax.set_ylabel(p1)
ax.set_yticks(np.arange(len(p1_vals)), labels=p1_vals)
ax.set_xlabel(p2)
ax.set_xticks(np.arange(len(p2_vals)), labels=p2_vals)
ax.set_title("Dimensionality")
fig.colorbar(im, ax=ax, shrink=0.4)

# average test/train scores vs. dimensionality
ax = fig.add_subplot(grid[1, 1])
ax.scatter(test_score_mean[:, :].flatten(), dim_mean[:, :].flatten())
ax.set_xlabel("score")
ax.set_ylabel("dim")
ax.set_title("Test score vs. dimensionality")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/ir_{p1}_{p2}.pdf')
plt.show()
