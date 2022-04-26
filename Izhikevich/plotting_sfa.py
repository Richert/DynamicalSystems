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
a = PyAuto.from_file(f"results/sfa.pkl", auto_dir=auto_dir)
ds = a.additional_attributes['d']
n = len(ds)

# load simulation data
fre_low = pickle.load(open(f"results/sfa_fre_low.p", "rb"))['results']
fre_high = pickle.load(open(f"results/sfa_fre_high.p", "rb"))['results']
rnn_low = pickle.load(open(f"results/sfa_rnn_low.p", "rb"))['results']
rnn_high = pickle.load(open(f"results/sfa_rnn_high.p", "rb"))['results']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (9, 6)
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
grid = gridspec.GridSpec(nrows=4, ncols=6, figure=fig)

# plot the 2D bifurcation diagram
ax = fig.add_subplot(grid[:2, :2])
a.plot_continuation('PAR(16)', 'PAR(19)', cont='d/I:lp1', ax=ax, line_color_stable='#5D6D7E')
a.plot_continuation('PAR(16)', 'PAR(19)', cont='d/I:lp2', ax=ax, line_color_stable='#5D6D7E')
a.plot_continuation('PAR(16)', 'PAR(19)', cont='d/I:hb1', ax=ax, line_color_stable='#148F77')
a.plot_continuation('PAR(16)', 'PAR(19)', cont='d/I:hb2', ax=ax, line_color_stable='#148F77')
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$d$')
ax.set_ylim([10.0, 110.0])
ax.set_xlim([10.0, 80.0])

# plot the 1D bifurcation diagrams
for j in range(0, n):
    ax = fig.add_subplot(grid[j, 2:])
    a.plot_continuation('PAR(16)', 'U(1)', cont=f'I:{j+1}', ax=ax, line_color_stable='#76448A',
                        line_color_unstable='#5D6D7E')
    try:
        a.plot_continuation('PAR(16)', 'U(1)', cont=f'I:{j+1}:lc', ax=ax, line_color_stable='#148F77', ignore=['BP'])
    except KeyError:
        pass
    ax.set_ylabel(r'$r$')
    ax.set_title(rf'$d = {ds[j]}$')
    ax.set_xlabel(R'$I$' if j == n-1 else '')
    ax.set_xlim([20.0, 80.0])

# plot the time signals
data = [[fre_low, rnn_low], [fre_high, rnn_high]]
titles = [rf'$d = {ds[0]}$', rf'$d = {ds[1]}$']
for i, ((fre, rnn), title) in enumerate(zip(data, titles)):
    ax = fig.add_subplot(grid[2, i*3:(i+1)*3])
    ax.plot(fre.index, rnn['r'])
    ax.plot(fre['r'])
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('r')
    ax.set_title(title)
    ax = fig.add_subplot(grid[3, i * 3:(i + 1) * 3])
    ax.plot(fre.index, rnn['u'])
    ax.plot(fre['u'])
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('u')

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.01, hspace=0.05, wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/sfa.pdf')
plt.show()
