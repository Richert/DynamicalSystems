from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyauto import PyAuto
import sys
import numpy as np
import pickle
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/eic_corrected.pkl", auto_dir=auto_dir)
deltas = a.additional_attributes['deltas']

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
markersize = 6


############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=3, ncols=3, figure=fig)

# 2D continuations
##################

ax = fig.add_subplot(grid[:, 0])
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lc1', ax=ax, line_style_unstable='solid')
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lc2', ax=ax, line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{fs}$')
ax.set_xlabel(r'$I_{fs}$')
ax.set_ylim([0.0, 1.6])
ax.set_xlim([0.0, 80.0])

# 1D continuations
#############

ax = fig.add_subplot(grid[0, 1:])
a.plot_continuation('PAR(36)', 'U(1)', cont=f'I_fs:1', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(36)', 'U(1)', cont='I_fs:1:lc1', ax=ax, line_color_stable='#148F77')
ax.set_xlabel(r'$I_{fs}$')
ax.set_ylabel(r'$r_{rs}$')
ax.set_xlim([0.0, 80.0])

ax = fig.add_subplot(grid[1, 1:])
a.plot_continuation('PAR(36)', 'U(1)', cont=f'I_fs:2', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(36)', 'U(1)', cont='I_fs:2:lc1', ax=ax, line_color_stable='#148F77')
ax.set_xlabel(r'$I_{fs}$')
ax.set_ylabel(r'$r_{rs}$')
ax.set_xlim([0.0, 80.0])

ax = fig.add_subplot(grid[2, 1:])
a.plot_continuation('PAR(36)', 'U(1)', cont=f'I_fs:3', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(36)', 'U(1)', cont='I_fs:3:lc1', ax=ax, line_color_stable='#148F77')
ax.set_xlabel(r'$I_{fs}$')
ax.set_ylabel(r'$r_{rs}$')
ax.set_xlim([0.0, 80.0])

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/eic_corrected.pdf')
plt.show()
