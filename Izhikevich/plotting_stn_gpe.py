from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from pyauto import PyAuto
import sys
import pickle
import numpy as np
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/stn_gpe.pkl", auto_dir=auto_dir)

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# 1D continuations
##################

# 1D continuation in I_s
ax = fig.add_subplot(grid[0, 0])
a.plot_continuation('PAR(16)', 'U(1)', cont='I_s:2', ax=ax, line_color_stable='#76448A', line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(16)', 'U(1)', cont='I_s:2:lc', ax=ax, line_color_stable='#148F77')
ax.set_xlabel(r'$I_s$')
ax.set_ylabel(r'$r_s$')

# 2D continuations
##################

# continuation of Hopf in I_s and Delta_g
ax = fig.add_subplot(grid[0, 1])
a.plot_continuation('PAR(16)', 'PAR(27)', cont=f'D_g/I_s:hb1', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#76448A', line_style_unstable='solid')
ax.set_xlabel(r'$I_s$')
ax.set_ylabel(r'$\Delta_g$')

# continuation of Hopf in both Deltas
ax = fig.add_subplot(grid[1, 0])
a.plot_continuation('PAR(27)', 'PAR(6)', cont=f'D_s/D_g:hb1', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#76448A', line_style_unstable='solid')
ax.set_xlabel(r'$\Delta_g$')
ax.set_ylabel(r'$\Delta_s$')

# continuation of Hopf in Delta and w
ax = fig.add_subplot(grid[1, 1])
a.plot_continuation('PAR(27)', 'PAR(42)', cont=f'w/D_g:hb1', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#76448A', line_style_unstable='solid')
ax.set_xlabel(r'$\Delta_g$')
ax.set_ylabel(r'$w$')

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/stn_gpe.pdf')
plt.show()
