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
plt.rcParams['figure.figsize'] = (8, 8)
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
p2 = "p"

# collect data of interest in dataframe
path = "results/ir_results"
fn = "impulse_response_rs"
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
            data_tmp.loc[0, "K_diff"] = file["K_diff"][idx]
            data_tmp.loc[0, p1] = file["sweep"][p1]
            data_tmp.loc[0, p2] = file["sweep"][p2]
            data = pd.concat([data, data_tmp], axis=0)

# prepare data for plotting
alphas = np.unique(data[p1].values)
ps = np.unique(data[p2].values)
n = len(alphas)
m = len(ps)
test_score_mean = np.zeros((n, m))
train_score_mean = np.zeros_like(test_score_mean)
dim_mean = np.zeros_like(test_score_mean)
kernel_diff_mean = np.zeros_like(test_score_mean)
for row in range(n):
    for col in range(m):
        alpha, p = alphas[row], ps[col]
        idx1 = data.loc[:, p1] == alpha
        idx2 = data.loc[:, p2] == p
        idx = np.argwhere(idx1.values * idx2.values == 1.0)[0]
        point = data.iloc[idx, :]
        test_score_mean[row, col] = point["test_score"]
        train_score_mean[row, col] = point["train_score"]
        dim_mean[row, col] = point["dimensionality"]
        kernel_diff_mean[row, col] = point["K_diff"]

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

# test scores
ax = fig.add_subplot(grid[0, 0])
im = ax.imshow(test_score_mean[1:, :])
ax.set_ylabel(p1)
ax.set_yticks(np.arange(len(alphas)), labels=alphas)
ax.set_xlabel(p2)
ax.set_xticks(np.arange(len(ps)), labels=ps)
ax.set_title("Test Scores")
plt.colorbar(im, ax=ax)

# training scores
ax = fig.add_subplot(grid[0, 1])
im = ax.imshow(train_score_mean[1:, :])
ax.set_ylabel(p1)
ax.set_yticks(np.arange(len(alphas)), labels=alphas)
ax.set_xlabel(p2)
ax.set_xticks(np.arange(len(ps)), labels=ps)
ax.set_title("Training Scores")
plt.colorbar(im, ax=ax)

# dimensionality
ax = fig.add_subplot(grid[1, 0])
im = ax.imshow(dim_mean[1:, :])
ax.set_ylabel(p1)
ax.set_yticks(np.arange(len(alphas)), labels=alphas)
ax.set_xlabel(p2)
ax.set_xticks(np.arange(len(ps)), labels=ps)
ax.set_title("Dimensionality")
plt.colorbar(im, ax=ax)

# Kernel difference
ax = fig.add_subplot(grid[1, 1])
im = ax.imshow(kernel_diff_mean[1:, :])
ax.set_ylabel(p1)
ax.set_yticks(np.arange(len(alphas)), labels=alphas)
ax.set_xlabel(p2)
ax.set_xticks(np.arange(len(ps)), labels=ps)
ax.set_title("Kernel Difference from Identity")
plt.colorbar(im, ax=ax)

# average test/train scores vs. dimensionality
ax = fig.add_subplot(grid[2, 0])
ax.plot(test_score_mean[1:, :].flatten(), dim_mean[1:, :].flatten(), "*-")
ax.set_xlabel("score")
ax.set_ylabel("dim")
ax.set_title("Test score vs. dimensionality")

# average test/train scores vs. kernel difference
ax = fig.add_subplot(grid[2, 1])
ax.plot(test_score_mean[1:, :].flatten(), kernel_diff_mean[1:, :].flatten(), "*-")
ax.set_xlabel("score")
ax.set_ylabel("diff")
ax.set_title("Test score vs. difference between kernel and identity")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/ir.pdf')
plt.show()
