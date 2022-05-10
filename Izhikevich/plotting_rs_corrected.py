from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from pyauto import PyAuto
import sys
import numpy as np
sys.path.append('../')
import pickle

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/rs_corrected.pkl", auto_dir=auto_dir)
v_reset = a.additional_attributes['v_reset']
n = len(v_reset)

# load simulation data
fre_low = pickle.load(open(f"results/spike_mech_fre.p", "rb"))['results']
fre_high = pickle.load(open(f"results/spike_mech_fre_inf.p", "rb"))['results']
rnn_low = pickle.load(open(f"results/spike_mech_rnn.p", "rb"))['results']
rnn_high = pickle.load(open(f"results/spike_mech_rnn_inf.p", "rb"))['results']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6
cmap = plt.get_cmap('copper', lut=n)

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=4, figure=fig)

# plot the 2D bifurcation diagram
ax = fig.add_subplot(grid[:, 0])
a.plot_continuation('PAR(16)', 'PAR(8)', cont=f'v_0/I:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(16)', 'PAR(8)', cont=f'v_0/I:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$v_0$')
ax.set_title('(A) 2D bifurcation diagram')
ax.set_xlim([-20.0, 60.0])
ax.set_ylim([-160.0, -60.0])

# plot the 1D bifurcation diagrams
ax = fig.add_subplot(grid[:, 1])
lines = []
for j in range(1, n + 1):
    c = to_hex(cmap(j, alpha=1.0))
    line = a.plot_continuation('PAR(16)', 'U(4)', cont=f'I:{j}', ax=ax, line_color_stable=c, line_color_unstable=c)
    lines.append(line)
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$s$')
ax.set_title('Spike-reset correction')
ax.set_xlim([-20.0, 60.0])
plt.legend(handles=lines, labels=[fr'$v_0 = {v}$' for v in v_reset], loc=2)

# plot the time signals
data = [[fre_low, rnn_low], [fre_high, rnn_high]]
titles = [rf'$v_0 = -70$', rf'$v_0 = -160$']
time = np.linspace(500.0, 4500.0, num=fre_low['s'].shape[0])
for i, ((fre, rnn), title) in enumerate(zip(data, titles)):
    ax = fig.add_subplot(grid[i, 2:])
    ax.plot(time, rnn['s'])
    ax.plot(time, fre['s'])
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('s')
    ax.set_title(title)

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.01, hspace=0.05, wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/rs_corrected.pdf')
plt.show()
