from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import numpy as np
from pyrates.utility.pyauto import PyAuto
import sys
sys.path.append('../../')

# preparations
##############

fname = 'stn_gpe_3pop_nosyns'

# load matlab variables
# data = pd.DataFrame.from_csv(f'results/{fname}.csv')
# params = pd.DataFrame.from_csv(f'results/{fname}_params.csv')

# load python data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/{fname}_conts.pkl", auto_dir=auto_dir)

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (5.25, 4.0)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.7
markersize = 40

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=5, ncols=8, figure=fig)

# 2d: k_stn x k_gp
ax = fig.add_subplot(grid[:3, :3])
ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'k_gp/k_stn:lp1', ax=ax, line_color_stable='#3689c9',
                         line_color_unstable='#3689c9', default_size=markersize,
                         line_style_unstable='solid', ignore=['UZ'])
ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'k_gp/k_stn:hb1', ax=ax, line_color_stable='#148F77',
                         line_color_unstable='#148F77', default_size=markersize,
                         line_style_unstable='solid', ignore=['UZ'])
ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'k_gp/k_stn:hb2', ax=ax, line_color_stable='#ee2b2b',
                         line_color_unstable='#ee2b2b', default_size=markersize,
                         line_style_unstable='solid', ignore=['UZ'])
ax.set_ylabel(r'$k_{stn}$')
ax.set_xlabel(r'$k_{gp}$')
ax.set_xlim([0.0, 30.0])
ax.set_ylim([0.0, 7.0])

# 1D continuation in k_stn for k_gp = 5.0
ax = fig.add_subplot(grid[0, 3:])
ax = a.plot_continuation('PAR(25)', 'U(3)', cont=f'k_stn:1', ax=ax, default_size=markersize, ignore=['UZ'])
ax = a.plot_continuation('PAR(25)', 'U(3)', cont=f'k_stn:1:lc1', ax=ax, default_size=markersize, ignore=['UZ'],
                         line_color_stable='#148F77', line_color_unstable='#148F77')
ax.set_xlabel('$k_{stn}$')

# 1D continuation in k_stn for k_gp = 5.0
ax = fig.add_subplot(grid[1, 3:])
ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'k_gp:1', ax=ax, default_size=markersize, ignore=['UZ'])
ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'k_gp:1:lc1', ax=ax, default_size=markersize, ignore=['UZ'],
                         line_color_stable='#148F77', line_color_unstable='#148F77')
ax.set_xlabel('$k_{gp}$')
#ax.set_ylabel('GPe-p firing rate')

# 1D continuation in k_stn for k_gp = 5.0
ax = fig.add_subplot(grid[2, 3:])
ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'k_gp:2', ax=ax, default_size=markersize, ignore=['UZ'])
ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'k_gp:2:lc1', ax=ax, default_size=markersize, ignore=['UZ'],
                         line_color_stable='#148F77', line_color_unstable='#148F77')
ax.set_xlabel('$k_{gp}$')

# padding
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)

fig.canvas.draw()
plt.savefig(f'{fname}_bf.svg')
plt.show()
