import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from pycobi import ODESystem
sys.path.append('../')


#############
# load data #
#############

rs = ODESystem.from_file(f"results/rs.pkl", auto_dir="~/PycharmProjects/auto-07p")
fs = ODESystem.from_file(f"results/fs.pkl", auto_dir="~/PycharmProjects/auto-07p")
lts = ODESystem.from_file(f"results/lts.pkl", auto_dir="~/PycharmProjects/auto-07p")

############
# plotting #
############

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

# create figure
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(ncols=2, nrows=2)

# plot rs bifurcation diagrams
ax = fig.add_subplot(grid[0, 0])
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:lp1", ignore=["UZ"], line_style_unstable="solid", ax=ax)
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:lp2", ignore=["UZ"], line_style_unstable="solid", ax=ax)
ax.set_xlabel(r"$I_{rs}$ (pA)")
ax.set_ylabel(r"$\Delta_{rs}$ (mV)")
ax.set_title(r"Regular Spiking Neurons: $\kappa_{rs} = 10$ pA")
ax.set_xlim([0, 80])
ax.set_ylim([0, 2.0])
ax.legend([])
ax = fig.add_subplot(grid[0, 1])
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:lp3", ignore=["UZ"], line_style_unstable="solid", ax=ax)
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:lp4", ignore=["UZ"], line_style_unstable="solid", ax=ax)
rs.plot_continuation("PAR(8)", "PAR(5)", cont="D/I:hb1", ignore=["UZ"], line_style_unstable="solid", ax=ax)
ax.set_xlabel(r"$I_{rs}$ (pA)")
ax.set_ylabel(r"$\Delta_{rs}$ (mV)")
ax.set_title(r"Regular Spiking Neurons: $\kappa_{rs} = 100$ pA")
ax.set_xlim([30, 80])
ax.set_ylim([0, 2.0])

# plot fs bifurcation diagram
ax = fig.add_subplot(grid[1, 0])
fs.plot_continuation("PAR(16)", "PAR(6)", cont="D/I:hb1", ignore=["UZ"], line_style_unstable="solid", ax=ax)
ax.set_xlabel(r"$I_{fs}$ (pA)")
ax.set_ylabel(r"$\Delta_{fs}$ (mV)")
ax.set_title(r"Fast Spiking Neurons")
ax.set_ylim([0, 1.0])
ax.set_xlim([0, 200])

# plot lts bifurcation diagram
ax = fig.add_subplot(grid[1, 1])
lts.plot_continuation("PAR(16)", "PAR(6)", cont="D/I:hb1", ignore=["UZ"], line_style_unstable="solid", ax=ax)
ax.set_xlabel(r"$I_{lts}$ (pA)")
ax.set_ylabel(r"$\Delta_{lts}$ (mV)")
ax.set_title(r"Low Threshold Spiking Neurons")
ax.set_ylim([0, 1.0])
ax.set_xlim([100, 300])

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/snn_bifurcations.svg')
plt.show()
