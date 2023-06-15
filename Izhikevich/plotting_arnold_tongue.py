import matplotlib.pyplot as plt
import pickle
import seaborn as sb

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
ticks = 5

# conditions to loop over
#########################

fig = plt.figure(figsize=(6, 6))
grid = fig.add_gridspec(2, 2)

# define conditions
path = "results"
files = ["fre_arnold_tongue", "fre_at_hom", "fre_at_het"]
titles = [r"(A) Coherence for $\alpha = 1.0$", "(B) Coherence for $\Delta_{rs} = 0.1$",
          "(C) Coherence for $\Delta_{rs} = 1.0$"]
ylabels = [r"$\Delta_{rs}$ (mV)", r"$\alpha$ (pA)", r"$\alpha$ (pA)"]
grids = [grid[0, :], grid[1, 0], grid[1, 1]]

for idx, (f, title, ylabel, g) in enumerate(zip(files, titles, ylabels, grids)):

    # load data
    data = pickle.load(open(f"{path}/{f}.pkl", "rb"))
    coh = data["coherence"]

    # plot coherence
    ax = fig.add_subplot(g)
    sb.heatmap(coh, vmin=0.0, vmax=1.0, ax=ax, annot=False, cbar=True if idx == 0 else False,
               xticklabels=ticks, yticklabels=ticks)
    ax.set_xlabel(r'$\omega$ (Hz)')
    ax.set_ylabel(ylabel)
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
