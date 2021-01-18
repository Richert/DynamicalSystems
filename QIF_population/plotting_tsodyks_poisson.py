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
data = [loadmat(f'data/tsodyks_allscales_v{i}.mat') for i in range(1, 5)]

# load python data
a = PyAuto.from_file('results/tsodyks_poisson.pkl', auto_dir='~/PycharmProjects/auto-07p')

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (7, 8)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 0.7
markersize = 40

# spike raster plot indices
N = 10000
cutoff = int(N * 0.05)
n_neurons = 50
idx = np.random.randint(cutoff, N-cutoff, n_neurons)

# continuation specific variables
conditions = [r'\textbf{A:} $U_0 = 0.2$, $\alpha = 0.0$, $\Delta = 0.4$',
              r'\textbf{B:} $U_0 = 0.2$, $\alpha = 0.0$, $\Delta = 0.01$',
              r'\textbf{C:} $U_0 = 1.0$, $\alpha = 0.04$, $\Delta = 0.4$',
              r'\textbf{D:} $U_0 = 1.0$, $\alpha = 0.04$, $\Delta = 0.01$']
etas = ['eta_3', 'eta_4', 'eta_1', 'eta_2']
eta_pos = [-0.6, -0.4, -0.75, -1.0]
xlims = [[-1.1, 0.2], [-1.1, 0.2], [-1.1, -0.7], [-1.1, 2.0]]
ylims = [[0.0, 0.8], [0.0, 0.8], [0.0, 0.5], [0.0, 2.0]]
stim_times = [[50.0, 170.0], [50.0, 170.0], [40.0], [40.0]]
stim_durs = [[60.0, 60.0], [60.0, 60.0], [200.0], [200.0]]
stim_strengths = [[0.5, -0.5], [0.5, -0.5], [-0.1], [1.0]]

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=8, ncols=3, figure=fig)

# start plotting
for i in range(4):

    # plot 1D continuation in eta
    ax1 = fig.add_subplot(grid[2*i:2*(i+1), 0])
    ax1 = a.plot_continuation('PAR(1)', 'U(1)', cont=etas[i], ax=ax1, default_size=markersize)
    if i > 1:
        ax1 = a.plot_continuation('PAR(1)', 'U(1)', cont=f"{etas[i]}:lc", ax=ax1, default_size=markersize,
                                  ignore=['BP'], line_color_stable='#7f7f7f', line_color_unstable='#7f7f7f')
    plt.axvline(x=eta_pos[i], color='b', linestyle='--')
    ax1.set_ylabel(r'$r$')
    ax1.set_xlabel('')
    ax1.set_xlim(xlims[i])
    ax1.set_ylim(ylims[i])

    # plot firing rates
    ax2 = fig.add_subplot(grid[2*i, 1:])
    ax2.plot(data[i]['times'].squeeze(), data[i]['r_rec_av'].squeeze(), color='k')
    ax2.plot(data[i]['times'].squeeze(), data[i]['r_mes_rec_av'].squeeze(), color='tab:purple')
    ax2.plot(data[i]['times'].squeeze(), data[i]['r_poisson_rec_av'].squeeze(), color='tab:orange')
    ax_lims = ax2.get_ylim()
    for t, d, eta in zip(stim_times[i], stim_durs[i], stim_strengths[i]):
        r = Rectangle((t, ax_lims[0]), width=d, height=ax_lims[1] - ax_lims[0], alpha=0.2, color='#7f7f7f')
        ax2.add_patch(r)
        rx, ry = r.get_xy()
        cx = rx + r.get_width() * 0.5
        cy = ry + r.get_height() * 0.2 if i == 0 and eta == 0.5 else ry + r.get_height() * 0.96
        ax2.annotate(rf'$I(t) = {eta}$', (cx, cy), color='k', weight='bold', fontsize=plt.rcParams['font.size'],
                     ha='center', va='top')
    ax2.set_ylabel(r'$r$')
    ax2.set_xlim([0, 280])

    # plot spikes
    ax3 = fig.add_subplot(grid[2*i+1, 1:])
    ax3.eventplot([data[i]['raster'][0, j][:, 0] if len(data[i]['raster'][0, j]) > 0 else np.asarray([])
                   for j in idx], colors='k')
    ax3.set_xlim([500000, 3300000])
    ax3.set_ylabel('neuron \#')
    ax3.set_xticklabels(['0', '50', '100', '150', '200', '250'])

    # plot legend
    if i == 1:
        leg = ax2.legend([r'$\mathrm{SNN}_{\mathrm{pre}}$', r'$\mathrm{SNN}_{\mathrm{post}}$',
                          r'$\mathrm{FRE}_{\mathrm{Poisson}}$'],
                         loc='center left', bbox_to_anchor=(0.78, 0.55), fontsize=plt.rcParams['font.size'])
        leg.set_in_layout(False)

    # condition title
    ax2.set_title(conditions[i])

ax1.set_xlabel(r'$\bar \eta$')
ax1.set_xticks([-1.0, 0.0, 1.0, 2.0])
ax3.set_xlabel('time')

# padding
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)

fig.canvas.draw()
fig.savefig('meanfield_bf.pdf')
plt.show()
