from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyauto import PyAuto
import sys
import numpy as np
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/eic.pkl", auto_dir=auto_dir)

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 6)
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
grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# 2D continuations
##################

# continuation in Delta_rs and I_rs
ax = fig.add_subplot(grid[0, 0])
line = a.plot_continuation('PAR(18)', 'PAR(7)', cont=f'D_rs/I_rs:hb1', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
line = a.plot_continuation('PAR(18)', 'PAR(7)', cont=f'D_rs/I_rs:lp1', ax=ax, line_color_stable='#5D6D7E',
                           line_color_unstable='#5D6D7E', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=line_data[:, 1], y2=np.zeros_like(line_data[:, 0]), color='#5D6D7E',
                 alpha=0.5)
ax.set_ylabel(r'$\Delta_{rs}$')
ax.set_xlabel(r'$I_{rs}$')
ax.set_title('(A)')
ax.set_ylim([0.0, 5.0])
ax.set_xlim([10.0, 110.0])

# continuation in Delta_fs and I_rs
ax = fig.add_subplot(grid[0, 1])
line = a.plot_continuation('PAR(18)', 'PAR(30)', cont=f'D_fs/I_rs:lp2', ax=ax, line_color_stable='#5D6D7E',
                           line_color_unstable='#5D6D7E', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=line_data[:, 1], y2=np.zeros_like(line_data[:, 0]), color='#5D6D7E',
                 alpha=0.5)
line = a.plot_continuation('PAR(18)', 'PAR(30)', cont=f'D_fs/I_rs:lp1', ax=ax, line_color_stable='#5D6D7E',
                           line_color_unstable='#5D6D7E', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=line_data[:, 1], y2=np.zeros_like(line_data[:, 0]), color='#FFFFFF')
line = a.plot_continuation('PAR(18)', 'PAR(30)', cont=f'D_fs/I_rs:hb1', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
ax.set_ylabel(r'$\Delta_{fs}$')
ax.set_xlabel(r'$I_{rs}$')
ax.set_title('(B)')
ax.set_ylim([0.0, 5.0])
ax.set_xlim([10.0, 110.0])

# continuation in Delta_rs and I_fs
ax = fig.add_subplot(grid[1, 0])
line = a.plot_continuation('PAR(36)', 'PAR(7)', cont=f'D_rs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_betweenx(y=line_data[:, 1], x1=line_data[:, 0], x2=np.zeros_like(line_data[:, 0])+200.0, color='#148F77',
                 alpha=0.5)
line = a.plot_continuation('PAR(36)', 'PAR(7)', cont=f'D_rs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                           line_color_unstable='#5D6D7E', line_style_unstable='solid')
line_data = line.get_paths()[1].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 1]), y2=line_data[:, 1], color='#5D6D7E', alpha=0.5)
ax.set_ylabel(r'$\Delta_{rs}$')
ax.set_xlabel(r'$I_{fs}$')
ax.set_title('(C)')
ax.set_ylim([0.0, 2.0])
ax.set_xlim([40.0, 200.0])

# continuation in Delta_fs and I_fs
ax = fig.add_subplot(grid[1, 1])
line = a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                           line_color_unstable='#5D6D7E', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=line_data[:, 1], y2=np.zeros_like(line_data[:, 0]), color='#5D6D7E',
                 alpha=0.5)
line = a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                           line_color_unstable='#5D6D7E', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=line_data[:, 1], y2=np.zeros_like(line_data[:, 0]), color='#FFFFFF')
line = a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
ax.set_ylabel(r'$\Delta_{fs}$')
ax.set_xlabel(r'$I_{fs}$')
ax.set_title('(D)')
ax.set_ylim([0.0, 2.0])
ax.set_xlim([0.0, 100.0])

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/eic.pdf')
plt.show()
