import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib import gridspec
from pycobi import ODESystem
import sys
sys.path.append('../')


# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
rs = ODESystem.from_file(f"results/rs.pkl", auto_dir=auto_dir)
fs = ODESystem.from_file(f"results/fs.pkl", auto_dir=auto_dir)
deltas = rs.additional_attributes['D']
n = len(deltas)

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
grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# RS pop
########

# 1D bifurcation diagrams
ax = fig.add_subplot(grid[:, 0])
lines = []
for j in range(1, n + 1):
    c = to_hex(cmap(j, alpha=1.0))
    line = rs.plot_continuation('PAR(16)', 'U(1)', cont=f'I:{j}', ax=ax, line_color_stable=c, line_color_unstable=c)
    lines.append(line)
ax.set_xlim([0.0, 70.0])
ax.set_yticks([0.0, 0.01, 0.02, 0.03, 0.04])
ax.set_yticklabels(['0', '10', '20', '30', '40'])
ax.set_xlabel(r'$I$ (pA)')
ax.set_ylabel(r'$r$ (Hz)')
ax.set_title(r'(A) Regular-spiking population')
plt.legend(handles=lines, labels=[D for D in deltas], title=r"$\Delta_v$ (mV)", loc=2)

# FS pop
########

# 2D bifurcation diagram in I and D
ax = fig.add_subplot(grid[0, 1])
line = fs.plot_continuation('PAR(16)', 'PAR(6)', cont=f'D/I:hb1', ax=ax, line_color_stable='#148F77',
                            line_color_unstable='#148F77')
ax.axhline(y=0.5, color='black', linestyle='--')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
ax.set_xlabel(r'$I$ (pA)')
ax.set_ylabel(r'$\Delta_v$ (mV)')
ax.set_title('(B) Fast-spiking population')
ax.set_xlim([0.0, 200.0])
ax.set_ylim([0.0, 2.0])

# plot continuation in input current for different deltas
ax = fig.add_subplot(grid[1, 1])
fs.plot_continuation('PAR(16)', 'U(1)', cont=f'I:1', ax=ax)
line = fs.plot_continuation('PAR(16)', 'U(1)', cont=f'I:1:lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                            custom_bf_styles={'LP': {'marker': 'p'}})
y1 = line.get_paths()[0].vertices
y2 = line.get_paths()[1].vertices
plt.fill_between(x=y1[:, 0], y1=y1[:, 1], y2=y2[:, 1], color='#148F77', alpha=0.5)
ax.set_xlabel(r'$I$ (pA)')
ax.set_ylabel(r'$r$ (Hz)')
ax.set_ylim([-0.02, 0.3])
ax.set_xlim([0.0, 130.0])
ax.set_yticks([0.0, 0.1, 0.2, 0.3])
ax.set_yticklabels(['0', '100', '200', '300'])
ax.set_title(r'(C) Fast-spiking population')

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.01, hspace=0.05, wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/rs_fs.pdf')
plt.show()
