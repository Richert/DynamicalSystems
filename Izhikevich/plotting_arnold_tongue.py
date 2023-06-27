import matplotlib.pyplot as plt
import pickle
import seaborn as sb
import numpy as np

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

fig = plt.figure(figsize=(6, 6))
grid = fig.add_gridspec(2, 2)

# define conditions
path = "results"
files = ["fre_arnold_tongue", "fre_at_hom", "fre_at_het"]
titles = [r"(A) Coherence for $\alpha = 4.0$ pA", "(B) Coherence for $\Delta_{rs} = 0.1$ mV",
          "(C) Coherence for $\Delta_{rs} = 1.0$ mV"]
ylabels = [r"$\Delta_{rs}$ (mV)", r"$\alpha$ (pA)", r"$\alpha$ (pA)"]
xticks = [[0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 2.0, 4.0, 6.0], [0.0, 2.0, 4.0, 6.0]]
yticks = [[0.0, 0.3, 0.6, 0.9, 1.2], [1.0, 10**(1/3), 10**(2/3), 10.0], [1.0, 10**(1/3), 10**(2/3), 10.0]]
yticklabels = [[f"{t}" for t in yticks[0]], [r"$10^0$", r"$10^{\frac{1}{3}}$", r"$10^{\frac{2}{3}}$", r"$10^1$"],
               [r"$10^0$", r"$10^{\frac{1}{3}}$", r"$10^{\frac{2}{3}}$", r"$10^1$"]]
grids = [grid[0, :], grid[1, 0], grid[1, 1]]

for idx, (f, title, ylabel, g, xt, yt, ytl) in enumerate(
        zip(files, titles, ylabels, grids, xticks, yticks, yticklabels)
                                                         ):

    # load data
    data = pickle.load(open(f"{path}/{f}.pkl", "rb"))
    coh = data["coherence"]

    # plot coherence
    ax = fig.add_subplot(g)
    sb.heatmap(coh, vmin=0.0, vmax=1.0, ax=ax, annot=False, cbar=True if idx == 0 else False)
    ax.set_xlabel(r'$\omega$ (Hz)')
    ax.set_ylabel(ylabel)
    ax.set_xticks([np.argmin(np.abs(coh.columns.values - t)) for t in xt], labels=[f"{t}" for t in xt], rotation=0)
    ax.set_yticks([np.argmin(np.abs(coh.index.values - t)) for t in yt], labels=ytl, rotation=0)
    ax.set_title(title)
    ax.invert_yaxis()

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/arnold_tongue.svg')
plt.show()
