from matplotlib import gridspec
import matplotlib.pyplot as plt
from pycobi import ODESystem
import sys
sys.path.append('../')
import pickle
import numpy as np

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = ODESystem.from_file(f"results/ik_biexp_bifs.pkl", auto_dir=auto_dir)

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=3, figure=fig)

# 2D continuation
#################

# settings
y_params = [("g", 4), ("d", 16), ("g", 4), ("g", 4), ("g", 4)]
x_params = [("eta", 8), ("eta", 8), ("d", 16), ("tau_r", 17), ("tau_d", 18)]
grid_locs = [grid[0, 0], grid[0, 1], grid[0, 2], grid[1, 0], grid[1, 1]]
bfs = ["lp1", "lp2", "hb1"]
colors = ['#5D6D7E', '#5D6D7E', '#148F77']

# plotting
for (x_key, x_idx), (y_key, y_idx), loc in zip(x_params, y_params, grid_locs):

    ax = fig.add_subplot(loc)
    for bf, c in zip(bfs, colors):
        try:
            a.plot_continuation(f'PAR({x_idx})', f'PAR({y_idx})', cont=f'{y_key}/{x_key}:{bf}', ax=ax,
                                line_color_stable=c, line_color_unstable=c)
        except KeyError:
            try:
                a.plot_continuation(f'PAR({x_idx})', f'PAR({y_idx})', cont=f'{x_key}/{y_key}:{bf}', ax=ax,
                                    line_color_stable=c, line_color_unstable=c)
            except KeyError:
                pass
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)

# 1D continuations
##################

# settings
params = [("d", 16, 1)]
var = "U(4)"
grid_locs = [grid[1, 2]]

# plotting
for (param, idx, cont), loc in zip(params, grid_locs):

    ax = fig.add_subplot(loc)
    try:
        a.plot_continuation(f'PAR({idx})', var, cont=f'{param}:{cont}', ax=ax)
    except KeyError:
        pass
    ax.set_xlabel(param)
    ax.set_ylabel(var)

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/ik_biexp_bifs.pdf')
plt.show()
