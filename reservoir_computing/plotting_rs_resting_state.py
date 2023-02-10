import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pickle
from pandas import DataFrame


def correlation(s: np.ndarray):
    s = s - np.mean(s)
    s = s / np.std(s)
    c = np.corrcoef(s, rowvar=False)
    c[np.isnan(c)] = 0.0
    c[np.eye(c.shape[0]) > 0.0] = 0.0
    return c


def get_dim(s: np.ndarray):
    s = s - np.mean(s)
    s = s / np.std(s)
    cov = s.T @ s
    cov[np.eye(cov.shape[0]) > 0] = 0.0
    eigs = np.abs(np.linalg.eigvals(cov))
    return np.sum(eigs)**2/np.sum(eigs**2)


# load data
###########

n_files = 6
p1 = "Delta"
p2 = "p"
dimensionality = {p1: [], p2: [], "dim": []}
results = {p1: [], p2: [], "s": []}
connectivity = {p1: [], p2: [], "C": []}

for id in range(1, n_files+1):

    res = pickle.load(open(f"results/rs_resting_state_{id}.pkl", "rb"))
    key = tuple([(key, val) for key, val in res["condition"].items()])
    for key in [p1, p2]:
        results[key].append(float(res["condition"][key]))
        dimensionality[key].append(float(res["condition"][key]))
        connectivity[key].append(float(res["condition"][key]))
    results["s"].append(res["s"])
    dimensionality["dim"].append(get_dim(res["s"].values))
    connectivity["C"].append(correlation(res["s"].values))

dimensionality = DataFrame.from_dict(dimensionality)
results = DataFrame.from_dict(results)
connectivity = DataFrame.from_dict(connectivity)

# plotting
##########

fig = plt.figure(figsize=(12, 8))
grid = GridSpec(nrows=2, ncols=2, figure=fig)

# spike raster plot
ax = fig.add_subplot(grid[0, :])
idx1 = np.argwhere(np.abs(dimensionality["Delta"].values - 1.0) < 1e-3).squeeze()
idx2 = np.argwhere(np.abs(dimensionality["Delta"].values - 0.1) < 1e-3).squeeze()
ax.bar(np.arange(1, len(idx1)+1)-0.25, height=dimensionality["dim"][idx1], color="blue", label=r"$\Delta = 1.0$",
       width=0.5)
ax.bar(np.arange(1, len(idx1)+1)+0.25, height=dimensionality["dim"][idx2], color="orange", label=r"$\Delta = 0.1$",
       width=0.5)
ax.set_xticks(np.arange(1, len(idx1)+1, step=1), labels=[dimensionality["p"][idx] for idx in idx1])
ax.set_xlabel("p")
ax.set_ylabel("dim")
plt.legend()

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/rs_resting_state.pdf')
plt.show()