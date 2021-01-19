from scipy.io import loadmat
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pyrates.utility.pyauto import PyAuto
import sys
sys.path.append('../')

# preparations
##############

# load matlab variables
data = loadmat(f'data/tsodyks_allscales.mat')

# load python data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
fname = 'results/tsodyks_mf.pkl'
fname2 = 'results/tsodyks_poisson.pkl'
a = PyAuto.from_file(fname, auto_dir=auto_dir)
a2 = PyAuto.from_file(fname2, auto_dir=auto_dir)

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (7, 3)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 0.7
markersize = 40

# spike raster plot indices
N = 10000
cutoff = int(N * 0.05)
n_neurons = 100
idx = np.random.randint(cutoff, N-cutoff, n_neurons)

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# 2D continuation in eta and Delta
##################################

ax = fig.add_subplot(grid[:, 0])
a.plot_continuation('PAR(1)', 'PAR(5)', cont='eta_Delta_hb1', ax=ax, line_style_unstable='solid', ignore=['GH'],
                    line_color_stable='#1f77b4', line_color_unstable='#1f77b4')
a2.plot_continuation('PAR(1)', 'PAR(7)', cont='eta_Delta_hb1', ax=ax, line_style_stable='dotted', ignore=['GH', 'LP'],
                     line_style_unstable='dotted', line_color_stable='#ff7f0e', line_color_unstable='#ff7f0e')
plt.arrow(x=-0.85, y=0.4, dx=0.23, dy=0, width=0.001, head_width=0.01, length_includes_head=True, color='k')
ax.annotate(rf'$I(t) = 0.23$', (-0.85, 0.38), color='k', weight='bold', fontsize=plt.rcParams['font.size'],
            ha='left', va='top')
ax.set_xlabel(r'$\bar \eta$')
ax.set_ylabel(r'$\Delta$')
ax.set_title(r'\textbf{A}')
ax.set_xlim([-1.0, 0.0])

# firing rate and spiking data
##############################

# firing rates
ax2 = fig.add_subplot(grid[0, 1])
ax2.plot(data['times'].squeeze(), data['r_rec_av'].squeeze(), color='k')
ax2.plot(data['times'].squeeze(), data['r_poisson_rec_av'].squeeze(), color='tab:orange')
ax2.plot(data['times'].squeeze(), data['r_mpa_rec_av'].squeeze(), color='tab:blue')
ax_lims = ax2.get_ylim()
r = Rectangle((250, ax_lims[0]), width=250, height=ax_lims[1] - ax_lims[0], alpha=0.2, color='#7f7f7f')
ax2.add_patch(r)
rx, ry = r.get_xy()
cx = rx + r.get_width() * 0.5
cy = ry + r.get_height() * 0.96
ax2.annotate(rf'$I(t) = 0.23$', (cx, cy), color='k', weight='bold', fontsize=plt.rcParams['font.size'],
             ha='center', va='top')
leg = ax2.legend([r'$\mathrm{SNN}_{\mathrm{pre}}$', r'$\mathrm{FRE}_{\mathrm{Poisson}}$',
                 r'$\mathrm{FRE}_{\mathrm{MPA}}$'],
                 loc='upper left', bbox_to_anchor=(0.05, 1.2), fontsize=plt.rcParams['font.size'])
leg.set_in_layout(False)
ax2.set_ylabel(r'$r$')
ax2.set_title(r'\textbf{B}')
ax2.set_xlim([0, 500])

# spikes
ax3 = fig.add_subplot(grid[1, 1])
ax3.eventplot([data['raster'][0, j][:, 0] if len(data['raster'][0, j]) > 0 else np.asarray([])
               for j in idx], colors='k')
ax3.set_xlim([200000, 5200000])
ax3.set_ylabel('neuron \#')
ax3.set_xlabel('time')
ax3.set_xticks([200000, 1200000, 2200000, 3200000, 4200000, 5200000])
ax3.set_xticklabels(['0', '100', '200', '300', '400', '500'])

# padding
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)

# save and show figure
fig.canvas.draw()
plt.savefig('mpa_bf.pdf')
plt.show()
