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
a = PyAuto.from_file(f"results/rs2.pkl", auto_dir=auto_dir)
v_reset = a.additional_attributes['v_reset']
n = len(v_reset)

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
grid = gridspec.GridSpec(nrows=1, ncols=1, figure=fig)

# plot the 1D bifurcation diagrams
ax = fig.add_subplot(grid[0, 0])
lines = []
for j in range(1, n + 1):
    c = to_hex(cmap(j, alpha=1.0))
    line = a.plot_continuation('PAR(16)', 'U(4)', cont=f'I:{j}', ax=ax, line_color_stable=c, line_color_unstable=c)
    lines.append(line)
# ax.set_xlim([0.0, 70.0])
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$s$')
ax.set_title('Spike-reset correction')
plt.legend(handles=lines, labels=[fr'$v_0 = {v}$' for v in v_reset], loc=2)

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.01, hspace=0.05, wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/rs2.pdf')
plt.show()
