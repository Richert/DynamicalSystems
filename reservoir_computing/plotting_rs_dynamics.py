import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import sys
import numpy as np


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

fname = f"rs_dc_{sys.argv[-1]}"
data = pickle.load(open(f"results/{fname}_results.pkl", "rb"))
config = pickle.load(open(f"config/{fname}_config.pkl", "rb"))
print(f"Condition: {config['sweep']}")

# plotting
##########

fig = plt.figure(figsize=(12, 8))
grid = GridSpec(nrows=2, ncols=2, figure=fig)

cutoff = 1000
signal = data["s"].values[cutoff:, :]

# spike raster plot
ax = fig.add_subplot(grid[0, :])
im = ax.imshow(signal.T, aspect=20.0, vmin=0.0, vmax=np.max(signal), interpolation="none")
ax.set_xlabel("time")
ax.set_ylabel("neurons")
plt.colorbar(im, ax=ax, shrink=0.5)
plt.title(f"Spiking activity")

# correlation
ax = fig.add_subplot(grid[1, 0])
c = correlation(signal)
im = ax.imshow(c, aspect=1.0, vmin=0.0, vmax=np.max(c), interpolation="none")
ax.set_xlabel("neurons")
ax.set_ylabel("neurons")
plt.colorbar(im, ax=ax, shrink=0.5)
plt.title(f"Correlation (dim = {get_dim(signal)})")

# average activity
ax = fig.add_subplot(grid[1, 1])
ax.plot(np.mean(signal, axis=-1))
ax.set_xlabel("time (ms)")
ax.set_ylabel("s")
ax.set_title("mean-field dynamics")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/{fname}_dynamics.pdf')
plt.show()
