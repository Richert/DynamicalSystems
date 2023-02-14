import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import numpy as np
import sys


# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6


# conditions to loop over
#########################

# define conditions
fn = sys.argv[-1]  #"rs_arnold_tongue"
conditions = ["het", "hom"]
titles = [r"$\Delta = 1.0$", r"$\Delta = 0.1$"]
base_len = 6
fig1 = plt.figure(1, figsize=(int(len(conditions)*base_len), base_len))
grid1 = GridSpec(ncols=len(conditions), nrows=1, figure=fig1)
fig2 = plt.figure(2, figsize=(int(len(conditions)*base_len), base_len))
grid2 = GridSpec(ncols=len(conditions), nrows=1, figure=fig2)

for idx, (cond, title) in enumerate(zip(conditions, titles)):

    # load data
    data = pickle.load(open(f"{fn}_{cond}.pkl", "rb"))
    alphas = data["alphas"]
    omegas = data["omegas"]
    coh = data["coherence"]
    plv = data["plv"]
    res_map = data["map"]

    # plot coherence
    ax = fig1.add_subplot(grid1[0, idx])
    cax = ax.imshow(coh[::-1, :], aspect='equal', interpolation="none")
    ax.set_xlabel(r'$\omega$ (Hz)')
    ax.set_ylabel(r'$\alpha$ (Hz)')
    ax.set_xticks(np.arange(0, len(omegas), 3))
    ax.set_yticks(np.arange(0, len(alphas), 3))
    ax.set_xticklabels(np.round(omegas[::3]*1e3, decimals=1))
    ax.set_yticklabels(np.round(alphas[::-3]*1e3, decimals=1))
    plt.title(f"Coh for {title}")
    if idx == len(conditions)-1:
        plt.colorbar(cax, ax=ax, shrink=0.5)

    # plot PLV
    ax = fig2.add_subplot(grid2[0, idx])
    cax = ax.imshow(plv[::-1, :], aspect='equal', interpolation="none")
    ax.set_xlabel(r'$\omega$ (Hz)')
    ax.set_ylabel(r'$\alpha$ (Hz)')
    ax.set_xticks(np.arange(0, len(omegas), 3))
    ax.set_yticks(np.arange(0, len(alphas), 3))
    ax.set_xticklabels(np.round(omegas[::3] * 1e3, decimals=1))
    ax.set_yticklabels(np.round(alphas[::-3] * 1e3, decimals=1))
    plt.title(f"PLV for {title}")
    if idx == len(conditions) - 1:
        plt.colorbar(cax, ax=ax, shrink=0.5)

# finishing touches
###################

# padding
fig1.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig2.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig1.canvas.draw()
plt.figure(fig1.number)
plt.savefig(f'results/rs_arnold_tongue_coh.pdf')
fig2.canvas.draw()
plt.figure(fig2.number)
plt.savefig(f'results/rs_arnold_tongue_plv.pdf')
plt.show()
