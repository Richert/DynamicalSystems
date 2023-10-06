import matplotlib.pyplot as plt
import sys
import numpy as np
import pickle
sys.path.append('../')

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

############
# plotting #
############

# create figure layout
fig = plt.figure(figsize=(12, 6), constrained_layout=True)
grid = fig.add_gridspec(4, 2)

# plot spike rasters
conditions = ["eic_het_low_sfa", "eic_het_high_sfa", "eic_hom_low_sfa", "eic_hom_high_sfa"]
titles = [
    r"(A) RS-FS: $\Delta_{fs} = 2.0$ mV, $\kappa_{rs} = 10.0$",
    r"(B) RS-FS: $\Delta_{fs} = 2.0$ mV, $\kappa_{rs} = 100.0$",
    r"(C) RS-FS: $\Delta_{fs} = 0.2$ mV, $\kappa_{rs} = 10.0$",
    r"(D) RS-FS: $\Delta_{fs} = 0.2$ mV, $\kappa_{rs} = 100.0$"
          ]
subplots = [[0, 0], [0, 1], [2, 0], [2, 1]]
ticks = [25, 15, 40, 30]
for cond, title, idx, ytick in zip(conditions, titles, subplots, ticks):
    data = pickle.load(open(f"results/snn_{cond}.p", "rb"))
    rs, fs = data["rs_spikes"], data["fs_spikes"]
    ax = fig.add_subplot(grid[idx[0], idx[1]])
    ax.imshow(rs.T, aspect="auto", cmap="Greys", interpolation="none")
    ax.set_title(title)
    ax.set_ylabel("RS neurons")
    ax = fig.add_subplot(grid[idx[0]+1, idx[1]])
    ax.imshow(fs.T, aspect="auto", cmap="Greys", interpolation="none")
    ax.set_ylabel("FS neurons")
    ax.set_xlabel("time (ms)")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/eic_snn_spikes.svg')
plt.show()
