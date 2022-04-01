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
a = PyAuto.from_file(f"results/izhikevich_fre.pkl", auto_dir=auto_dir)
deltas = a.additional_attributes['D']
n = len(deltas)

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (7, 8)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6
cmap = plt.get_cmap('copper', lut=n)

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=6, ncols=2, figure=fig)

# plot continuation in input current for different deltas
ax = fig.add_subplot(grid[0:2, 0])
lines = []
for i in range(1, n+1):
    c = to_hex(cmap(i, alpha=1.0))
    line = a.plot_continuation('PAR(16)', 'U(1)', cont=f'I:{i}', ax=ax, line_color_stable=c, line_color_unstable=c)
    lines.append(line)
ax.set_xlim([0.0, 70.0])
ax.set_ylim([0.0, 0.025])
ax.set_xlabel(r'$I$ (pA)')
ax.set_ylabel(r'$r$ (Hz)')
ax.set_yticks(ticks=ax.get_yticks(), labels=[f"{np.round(tick * 1e3, decimals=1)}" for tick in ax.get_yticks()])
ax.set_title('Fixed Points')
plt.legend(handles=lines, labels=[fr'$\Delta_v = {D}$' for D in deltas])

# 2D bifurcation diagram in I and D
ax = fig.add_subplot(grid[0:2, 1])
a.plot_continuation('PAR(16)', 'PAR(6)', cont=f'D/I:hb1', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#76448A')
a.plot_continuation('PAR(16)', 'PAR(6)', cont=f'D/I:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(16)', 'PAR(6)', cont=f'D/I:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E')
ax.set_xlabel(r'$I$ (pA)')
ax.set_ylabel(r'$\Delta_v$ (nS/mV)')

# plot I continuation for two different Deltas including the limit cycle
targets = a.additional_attributes['targets_1d']
for i, t in enumerate(targets):
    ax = fig.add_subplot(grid[2:4, i])
    a.plot_continuation('PAR(16)', 'U(1)', cont=f'I:{t+1}', ax=ax, line_color_stable='#76448A',
                        line_color_unstable='#5D6D7E', custom_bf_styles={'LP': {'marker': 'v'}})
    a.plot_continuation('PAR(16)', 'U(1)', cont=f'I:{t+1}:lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                        custom_bf_styles={'LP': {'marker': 'p'}})
    ax.set_xlim([25.0, 75.0])
    ax.set_ylim([0., 0.05])
    ax.set_xlabel(r'$I$ (pA)')
    ax.set_ylabel(r'$r$ (Hz)')
    ax.set_yticks(ticks=ax.get_yticks(), labels=[f"{np.round(tick * 1e3, decimals=1)}" for tick in ax.get_yticks()])
    ax.set_title(rf'$\Delta_v = {deltas[t]}$')

# plot firing rate dynamics
for i, t in enumerate(targets):
    ax = fig.add_subplot(grid[4, i])
    res = pickle.load(open(f"results/izhikevich_sim_exc{i+1}.p", "rb"))['results']
    ax.plot(res['r'])
    ax.set_xlabel('time (ms)')
    ax.set_ylabel(r'$r$ (Hz)')
    ax.set_yticks(ticks=ax.get_yticks(), labels=[f"{np.round(tick * 1e3, decimals=1)}" for tick in ax.get_yticks()])

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/izhikevich_bf_exc.pdf')
plt.show()
