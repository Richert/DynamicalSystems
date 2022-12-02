from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyauto import PyAuto
import sys
import pickle
import numpy as np
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/fs.pkl", auto_dir=auto_dir)
deltas = a.additional_attributes['D']
n = len(deltas)

# load simulation data
fre = pickle.load(open(f"results/fs_fre.p", "rb"))['results']
rnn = pickle.load(open(f"results/fs_rnn.p", "rb"))['results']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 4)
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
grid = gridspec.GridSpec(nrows=5, ncols=2, figure=fig)

# 2D continuation
#################

# 2D bifurcation diagram in I and D
ax = fig.add_subplot(grid[:3, 0])
line = a.plot_continuation('PAR(16)', 'PAR(6)', cont=f'D/I:hb1', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77')
ax.axhline(y=0.5, color='black', linestyle='--')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$\Delta_v$')
ax.set_title('(A) 2D bifurcation diagram')
ax.set_xlim([0.0, 300.0])
ax.set_ylim([0.0, 2.0])

# 1D continuations
##################

# plot continuation in input current for different deltas
ax = fig.add_subplot(grid[:3, 1])
a.plot_continuation('PAR(16)', 'U(4)', cont=f'I:1', ax=ax)
line = a.plot_continuation('PAR(16)', 'U(4)', cont=f'I:1:lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                           custom_bf_styles={'LP': {'marker': 'p'}})
ax.axvline(x=60.0, color='#5D6D7E', linestyle='--')
ax.axvline(x=120.0, color='#148F77', linestyle='--')
y1 = line.get_paths()[0].vertices
y2 = line.get_paths()[1].vertices
plt.fill_between(x=y1[:, 0], y1=y1[:, 1], y2=y2[:, 1], color='#148F77', alpha=0.5)
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$s$')
ax.set_ylim([-0.01, 0.9])
ax.set_xlim([0.0, 200.0])
ax.set_title(r'(B) 1D bifurcation diagram for $\Delta_v = 0.5$')

# time series plotting
######################

title = r'(C) Fast-spiking population dynamics for $\Delta = 0.5$'
ax = fig.add_subplot(grid[3:, :])
ax.plot(fre['s'])
ax.plot(fre.index, rnn['s'])
xmin = np.min(fre.v1)
xmax = np.max(fre.v1)
plt.fill_betweenx([xmin - 0.1 * xmax, xmax + 0.1 * xmax], x1=800, x2=1200.0, color='grey', alpha=0.15)
ax.set_ylim([xmin - 0.1 * xmax, xmax + 0.1 * xmax])
ax.set_ylabel(r'$s$')
ax.set_title(title)
plt.legend(['mean-field', 'spiking network'])
ax.set_xlabel(r'time (ms)')

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/fs.pdf')
plt.show()
