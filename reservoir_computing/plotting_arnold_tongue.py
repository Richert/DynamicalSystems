import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import numpy as np
import sys
import seaborn as sb
from pandas import DataFrame

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 5.5)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6


# conditions to loop over
#########################

# define conditions
fn = "results/mf_entrainment"  #sys.argv[-1]
conditions = ["fp", "lc"]
titles = ["Coherence (FP)", "Coherence (LC)"]
base_len = 6
fig1 = plt.figure(1, figsize=(int(len(conditions)*base_len), base_len))
grid1 = GridSpec(ncols=len(conditions)+1, nrows=1, figure=fig1)

for idx, (cond, title) in enumerate(zip(conditions, titles)):

    # load data
    data = pickle.load(open(f"{fn}_{cond}.pkl", "rb"))
    coh = data["coherence"]

    # plot coherence
    ax = fig1.add_subplot(grid1[0, idx])
    sb.heatmap(coh, vmin=0.0, vmax=1.0, ax=ax, annot=False,
               cbar=True if idx == len(conditions)-1 else False, xticklabels=3, yticklabels=3, square=True,
               cbar_kws={"shrink": 0.4})
    ax.set_xlabel(r'$\omega$ (Hz)')
    ax.set_ylabel(r'$\Delta$ (mV)')
    ax.set_title(title)

# finishing touches
###################

# padding
fig1.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig1.canvas.draw()
plt.savefig(f'results/rs_arnold_tongue.pdf')
plt.show()
