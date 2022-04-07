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
exc1 = PyAuto.from_file(f"results/rs.pkl", auto_dir=auto_dir)
exc2 = PyAuto.from_file(f"results/rs2.pkl", auto_dir=auto_dir)
deltas = exc1.additional_attributes['D']
n = len(deltas)

# load simulation data
fre_hom = pickle.load(open(f"results/rs_fre_hom.p", "rb"))['results']
fre_het = pickle.load(open(f"results/rs_fre_het.p", "rb"))['results']
fre2_hom = pickle.load(open(f"results/rs_fre2_hom.p", "rb"))['results']
fre2_het = pickle.load(open(f"results/rs_fre2_het.p", "rb"))['results']

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
grid = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)

# plot the 1D bifurcation diagrams
titles = [r'(A) $\Delta_v$', r'(B) $\Delta_{\eta}$']
autos = [exc1, exc2]
for i, (a, title) in enumerate(zip(autos, titles)):

    ax = fig.add_subplot(grid[:2, i])
    lines = []
    for j in range(1, n + 1):
        c = to_hex(cmap(j, alpha=1.0))
        line = a.plot_continuation('PAR(16)', 'U(4)', cont=f'I:{j}', ax=ax, line_color_stable=c, line_color_unstable=c)
        lines.append(line)
    ax.set_xlim([0.0, 70.0])
    ax.set_xlabel(r'$I$')
    ax.set_ylabel(r'$s$' if i == 0 else '')
    ax.set_title(title)
    if i == 1:
        plt.legend(handles=lines, labels=[fr'$\Delta = {D}$' for D in deltas], loc=7)

# plot the timeseries
titles = [r'(C) $\Delta = 0.2$', r'(D) $\Delta = 1.6$', ]
data = [[fre_hom, fre2_hom], [fre_het, fre2_het]]
for i, (title, (res1, res2)) in enumerate(zip(titles, data)):

    ax = fig.add_subplot(grid[i+2, :])
    ax.plot(res1['s'])
    ax.plot(res2['s'])
    ax.set_ylabel(r'$s$')
    ax.set_title(title)
    if i == 0:
        plt.legend([r'$\Delta_v$', r'$\Delta_{\eta}$'])
    if i == 1:
        ax.set_xlabel(r'time (ms)')

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.01, hspace=0.05, wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/rs.pdf')
plt.show()