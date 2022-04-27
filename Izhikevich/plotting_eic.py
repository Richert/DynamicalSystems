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
a = PyAuto.from_file(f"results/eic.pkl", auto_dir=auto_dir)
deltas = a.additional_attributes['deltas']

# load simulation data
fre_hom = pickle.load(open(f"results/eic_fre_hom.p", "rb"))['results']
fre_het = pickle.load(open(f"results/eic_fre_het.p", "rb"))['results']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 4.5)
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
grid = gridspec.GridSpec(nrows=3, ncols=6, figure=fig)

# 2D continuations
##################

# continuation in Delta_rs and I_fs
ax = fig.add_subplot(grid[:2, :2])
line = a.plot_continuation('PAR(36)', 'PAR(7)', cont=f'D_rs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 1]), y2=line_data[:, 1], color='#148F77',
                 alpha=0.5)
line = a.plot_continuation('PAR(36)', 'PAR(7)', cont=f'D_rs/I_fs:hb3', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_betweenx(y=line_data[:, 1], x1=line_data[:, 0], x2=np.zeros_like(line_data[:, 0])+100.0, color='#148F77',
                  alpha=0.5)
line = a.plot_continuation('PAR(36)', 'PAR(7)', cont=f'D_rs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                           line_color_unstable='#5D6D7E', line_style_unstable='solid')
line_data = line.get_paths()[1].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 1]), y2=line_data[:, 1], color='#5D6D7E', alpha=0.5)
ax.set_ylabel(r'$\Delta_{rs}$')
ax.set_xlabel(r'$I_{fs}$')
ax.set_title('(A)')
ax.set_ylim([0.0, 1.6])
ax.set_xlim([0.0, 100.0])

# continuation in Delta_fs and I_fs
ax = fig.add_subplot(grid[:2, 2:4])
line1 = a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable='solid')
line2 = a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable='solid')
l1 = line1.get_paths()[0].vertices
l2_tmp = line2.get_paths()[0].vertices
l2 = np.interp(l1[:, 1], l2_tmp[:, 1], l2_tmp[:, 0])
plt.fill_betweenx(y=l1[:, 1], x2=l1[:, 0], x1=l2, color='#5D6D7E', alpha=0.5)
line = a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
ax.set_ylabel(r'$\Delta_{fs}$')
ax.set_xlabel(r'$I_{fs}$')
ax.set_title('(B)')
ax.set_ylim([0.0, 1.6])
ax.set_xlim([0.0, 100.0])

# 1D continuations
##################

# continuation in FS input for high Delta_fs
ax = fig.add_subplot(grid[0, 4:6])
a.plot_continuation('PAR(36)', 'U(1)', cont='I_fs:2', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax.set_ylabel(r'$r_{rs}$')
ax.set_xlabel('')
ax.set_title(fr'(C) $\Delta = {deltas[1]}$')
ax.set_xlim([0.0, 100.0])

# continuation in FS input for low Delta_fs
ax = fig.add_subplot(grid[1, 4:6])
a.plot_continuation('PAR(36)', 'U(1)', cont='I_fs:1', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(36)', 'U(1)', cont='I_fs:1:lc1', ax=ax, line_color_stable='#148F77')
a.plot_continuation('PAR(36)', 'U(1)', cont='I_fs:1:lc2', ax=ax, line_color_stable='#148F77')
ax.set_xlabel(r'$I_{fs}$')
ax.set_ylabel(r'$r_{rs}$')
ax.set_title(fr'(D) $\Delta = {deltas[0]}$')
ax.set_xlim([0.0, 100.0])

# time series
#############

data = [fre_hom, fre_het]
titles = [fr'(E) $\Delta = {deltas[0]}$', fr'(F) $\Delta = {deltas[1]}$']
for i, (fre, title) in enumerate(zip(data, titles)):
    ax = fig.add_subplot(grid[2, i*3:(i+1)*3])
    ax.plot(fre)
    ax.set_xlabel('time (ms)')
    ax.set_title(title)
    if i == len(data)-1:
        plt.legend(fre.columns.values)
    elif i == 0:
        ax.set_ylabel(r'$r$')

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/eic.pdf')
plt.show()