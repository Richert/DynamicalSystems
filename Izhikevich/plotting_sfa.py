from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyauto import PyAuto
import sys
sys.path.append('../')
import pickle
import numpy as np

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/sfa.pkl", auto_dir=auto_dir)
ds = a.additional_attributes['d']
n = len(ds)

# load simulation data
fre_low = pickle.load(open(f"results/sfa_fre_low.p", "rb"))
fre_high = pickle.load(open(f"results/sfa_fre_high.p", "rb"))
rnn_low = pickle.load(open(f"results/sfa_rnn_low.p", "rb"))['results']
rnn_high = pickle.load(open(f"results/sfa_rnn_high.p", "rb"))['results']
rnn2_low = pickle.load(open(f"results/sfa_rnn2_low.p", "rb"))['results']
rnn2_high = pickle.load(open(f"results/sfa_rnn2_high.p", "rb"))['results']
rnn_bfs = pickle.load(open("results/sfa_results.p", "rb"))

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6
cmap = plt.get_cmap('copper', lut=n)

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=5, ncols=6, figure=fig)

# plot the 2D bifurcation diagram
n_points = 2
ax = fig.add_subplot(grid[:2, :3])
ax.scatter(rnn_bfs['lp1'][::n_points, 0], rnn_bfs['lp1'][::n_points, 1], c='#5D6D7E', marker='x', s=10, alpha=0.5)
ax.scatter(rnn_bfs['lp2'][::n_points, 0], rnn_bfs['lp2'][::n_points, 1], c='#5D6D7E', marker='x', s=10, alpha=0.5)
ax.scatter(rnn_bfs['hb1'][::n_points, 0], rnn_bfs['hb1'][::n_points, 1], c='#148F77', marker='x', s=10, alpha=0.5)
ax.scatter(rnn_bfs['hb2'][::n_points, 0], rnn_bfs['hb2'][::n_points, 1], c='#148F77', marker='x', s=10, alpha=0.5)
line1 = a.plot_continuation('PAR(16)', 'PAR(19)', cont='d/I:lp1', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable='solid')
line2 = a.plot_continuation('PAR(16)', 'PAR(19)', cont='d/I:lp2', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable='solid')
l1, l2 = line2.get_paths()[:2]
l3 = line1.get_paths()[0]
l1 = np.concatenate([l1.vertices, l2.vertices], axis=0)
l3 = np.interp(l1[:, 1], l3.vertices[:, 1], l3.vertices[:, 0])
plt.fill_betweenx(y=l1[:, 1], x2=l1[:, 0], x1=l3, color='#5D6D7E', alpha=0.5)
line3 = a.plot_continuation('PAR(16)', 'PAR(19)', cont='d/I:hb1', ax=ax, line_color_stable='#148F77')
line4 = a.plot_continuation('PAR(16)', 'PAR(19)', cont='d/I:hb2', ax=ax, line_color_stable='#148F77')
l3, l4_tmp = line3.get_paths()[0], line4.get_paths()[0]
l4 = np.interp(l3.vertices[:, 1], l4_tmp.vertices[:, 1], l4_tmp.vertices[:, 0])
plt.fill_betweenx(y=l3.vertices[:, 1], x2=l3.vertices[:, 0], x1=l4, color='#148F77', alpha=0.5)
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$d$')
ax.set_title('(A) 2D bifurcation diagram')
ax.set_ylim([10.0, 120.0])
ax.set_xlim([20.0, 75.0])

# plot the 1D bifurcation diagrams
titles = ['B', 'C']
for j in range(0, n):
    ax = fig.add_subplot(grid[j, 3:])
    a.plot_continuation('PAR(16)', 'U(4)', cont=f'I:{j+1}', ax=ax, line_color_stable='#76448A',
                        line_color_unstable='#5D6D7E')
    try:
        a.plot_continuation('PAR(16)', 'U(4)', cont=f'I:{j+1}:lc', ax=ax, line_color_stable='#148F77', ignore=['BP'])
    except KeyError:
        pass
    ax.set_ylabel(r'$s$')
    ax.set_title(rf'({titles[j]}) $d = {ds[j]}$')
    ax.set_xlabel(R'$I$' if j == n-1 else '')
    ax.set_xlim([20.0, 75.0])

# plot the time signals
data = [[fre_low, rnn_low, rnn2_low], [fre_high, rnn_high, rnn2_high]]
titles = [rf'(D) $d = {ds[0]}$', rf'(E) $d = {ds[1]}$']
for i, ((fre, rnn, rnn2), title) in enumerate(zip(data, titles)):
    ax = fig.add_subplot(grid[2, i*3:(i+1)*3])
    ax.plot(fre['results'].index, rnn['s'])
    ax.plot(fre['results'].index, rnn2['s'])
    ax.plot(fre['results']['s'])
    if i == 0:
        plt.legend([r'spiking network ($u_i$)', r'spiking network ($u$)', 'mean-field'], loc=2)
    ax.set_ylabel(r'$s$')
    ax.set_title(title)
    ax = fig.add_subplot(grid[3, i * 3:(i + 1) * 3])
    ax.plot(fre['results'].index, rnn['u'])
    ax.plot(fre['results'].index, rnn2['u'])
    ax.plot(fre['results']['u'])
    ax.set_ylabel(r'$u$')
    ax = fig.add_subplot(grid[4, i * 3:(i + 1) * 3])
    ax.plot(fre['results'].index, fre['inp'], c='grey')
    ax.set_ylabel(r'$I$')
    ax.set_xlabel('time (ms)')

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.01, hspace=0.05, wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/sfa.pdf')
plt.show()
