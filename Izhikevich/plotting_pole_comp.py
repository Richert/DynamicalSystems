from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyauto import PyAuto
import sys
import pickle
import numpy as np
sys.path.append('../')

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
s = 20
marker = "*"
alpha = 0.5
ls = 'dotted'

############
# plotting #
############

# conditions
conditions = ["rs", "eic1", "eic2"]
state_vars = ["r", "rs", "rs"]

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=len(conditions), ncols=1, figure=fig)

for i, (c, v) in enumerate(zip(conditions, state_vars)):

    ax = fig.add_subplot(grid[i])
    orig = pickle.load(open(f"results/pole_comp_{c}_orig.p", "rb"))['results']
    new = pickle.load(open(f"results/pole_comp_{c}_new.p", "rb"))['results']
    time = orig.index
    ax.plot(time, 1e3*orig[v], color="blue")
    ax.plot(time, 1e3*new[v], color="orange")
    ax.set_ylabel("r (Hz)")
    ax.set_xlabel("time (ms)")

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/pole_comp.pdf')
plt.show()
