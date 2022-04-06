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
a = PyAuto.from_file(f"results/ib.pkl", auto_dir=auto_dir)
deltas = a.additional_attributes['D']
n = len(deltas)

# load simulation data
fre_hom = pickle.load(open(f"results/ib_fre_hom.p", "rb"))['results']
fre_het = pickle.load(open(f"results/ib_fre_het.p", "rb"))['results']
rnn_hom = pickle.load(open(f"results/ib_rnn_hom.p", "rb"))['results']
rnn_het = pickle.load(open(f"results/ib_rnn_het.p", "rb"))['results']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 8)
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
grid = gridspec.GridSpec(nrows=4, ncols=3, figure=fig)

# 2D continuation
#################

# 2D bifurcation diagram in I and D
ax = fig.add_subplot(grid[:2, 0])
a.plot_continuation('PAR(16)', 'PAR(6)', cont=f'D/I:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77')
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$\Delta_v$')
ax.set_title('(A) 2D bifurcation diagram')

# 1D continuations
##################

# plot continuation in input current for different deltas
subtitles = ['B', 'C']
for i in range(0, n):
    ax = fig.add_subplot(grid[i, 1:])
    a.plot_continuation('PAR(16)', 'U(4)', cont=f'I:{i+1}', ax=ax, line_color_stable='#76448A',
                        line_color_unstable='#5D6D7E')
    a.plot_continuation('PAR(16)', 'U(4)', cont=f'I:{i+1}:lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                        custom_bf_styles={'LP': {'marker': 'p'}})
    ax.set_xlabel(r'$I$' if i == 1 else '')
    ax.set_ylabel(r'$s$')
    ax.set_ylim([-0.01, 0.4])
    ax.set_xlim([0.0, 600.0])
    ax.set_title(rf'({subtitles[i]}) 1D bifurcation diagram for $\Delta_v = {deltas[i]}$')

# time series plotting
######################

titles = [r'(D) $\Delta = 0.04$', r'(E) $\Delta = 0.12$', ]
data = [[fre_hom, rnn_hom], [fre_het, rnn_het]]
for i, (title, (fre, rnn)) in enumerate(zip(titles, data)):

    ax = fig.add_subplot(grid[i+2, :])
    ax.plot(fre['s'])
    ax.plot(fre.index, rnn['s'])
    ax.set_ylabel(r'$s$')
    ax.set_title(title)
    if i == 0:
        plt.legend(['FRE', 'RNN'])
    if i == 1:
        ax.set_xlabel(r'time (ms)')

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/ib.pdf')
plt.show()
