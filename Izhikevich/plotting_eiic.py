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
a = PyAuto.from_file(f"results/eiic.pkl", auto_dir=auto_dir)
deltas = a.additional_attributes['deltas']

# load simulation data
fre_hom = pickle.load(open(f"results/eiic_fre_hom.p", "rb"))['results']
fre_het = pickle.load(open(f"results/eiic_fre_het.p", "rb"))['results']

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

# continuation in Delta_lts and I_lts
ax = fig.add_subplot(grid[:2, :2])
line1 = a.plot_continuation('PAR(54)', 'PAR(48)', cont=f'D_lts/I_lts:lp1', ax=ax, line_color_stable='#5D6D7E',
                           line_color_unstable='#5D6D7E', line_style_unstable='solid')
line2 = a.plot_continuation('PAR(54)', 'PAR(48)', cont=f'D_lts/I_lts:lp2', ax=ax, line_color_stable='#5D6D7E',
                           line_color_unstable='#5D6D7E', line_style_unstable='solid')
x1 = line1.get_paths()[0].vertices
x2_tmp = line2.get_paths()[0].vertices
x2 = np.interp(x1[:, 1], x2_tmp[:, 1], x2_tmp[:, 0])
plt.fill_betweenx(y=x1[:, 1], x1=x1[:, 0], x2=x2, color='#5D6D7E', alpha=0.5)
line = a.plot_continuation('PAR(54)', 'PAR(48)', cont=f'D_lts/I_lts:hb2', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
ax.set_ylabel(r'$\Delta_{lts}$')
ax.set_xlabel(r'$I_{lts}$')
ax.set_title('(A)')
ax.set_ylim([0.0, 2.0])
ax.set_xlim([50.0, 150.0])

# continuation in Delta_lts and I_lts
ax = fig.add_subplot(grid[:2, 2:4])
line = a.plot_continuation('PAR(54)', 'PAR(30)', cont=f'D_fs/I_lts:lp1', ax=ax, line_color_stable='#5D6D7E',
                           line_color_unstable='#5D6D7E', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#5D6D7E', alpha=0.5)
line = a.plot_continuation('PAR(54)', 'PAR(30)', cont=f'D_fs/I_lts:lp2', ax=ax, line_color_stable='#5D6D7E',
                           line_color_unstable='#5D6D7E', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#FFFFFF')
line = a.plot_continuation('PAR(54)', 'PAR(30)', cont=f'D_fs/I_lts:hb2', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
ax.set_ylabel(r'$\Delta_{fs}$')
ax.set_xlabel(r'$I_{lts}$')
ax.set_title('(B)')
ax.set_ylim([0.0, 3.0])
ax.set_xlim([50.0, 150.0])

# 1D continuations
##################

# continuation in I_lts for low heterogeneity
ax = fig.add_subplot(grid[0, 4:])
a.plot_continuation('PAR(54)', 'U(1)', cont='I_lts:2', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax.set_xlabel(r'$I_{lts}$')
ax.set_ylabel(r'$r_{rs}$')
ax.set_title(fr'(C) $\Delta = {deltas[1]}$')
ax.set_xlim([50.0, 150.0])

# continuation in I_lts for low heterogeneity
ax = fig.add_subplot(grid[1, 4:])
a.plot_continuation('PAR(54)', 'U(1)', cont='I_lts:1', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(54)', 'U(1)', cont='I_lts:1:lc1', ax=ax, line_color_stable='#148F77', ignore=['BP'])
a.plot_continuation('PAR(54)', 'U(1)', cont='I_lts:1:lc2', ax=ax, line_color_stable='#148F77', ignore=['BP'])
ax.set_xlabel(r'$I_{lts}$')
ax.set_ylabel(r'$r_{rs}$')
ax.set_title(fr'(D) $\Delta = {deltas[0]}$')
ax.set_xlim([50.0, 150.0])

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
plt.savefig(f'results/eiic.pdf')
plt.show()
