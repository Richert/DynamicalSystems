from matplotlib import gridspec
import matplotlib.pyplot as plt
from pycobi import ODESystem
import sys
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = ODESystem.from_file(f"results/rs_bifs.pkl", auto_dir=auto_dir)

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
grid = gridspec.GridSpec(nrows=3, ncols=3, figure=fig)

# 2D continuation
#################

# settings
x_params = [("eta", 8), ("tau_s", 17), ("d", 16)]
y_params = [("g", 4), ("g", 4), ("g", 4)]
x_lims = [(0.0, 100.0), (0.0, 50.0), (0.0, 200.0)]
y_lims = [(0.0, 50.0), (0.0, 50.0), (0.0, 50.0)]
grid_locs = [grid[:2, 0], grid[:2, 1], grid[:2, 2]]
bfs = ["lp1", "lp2", "hb1", "hb2"]
colors = ['#5D6D7E', '#5D6D7E', '#148F77', '#148F77']

# plotting
for (x_key, x_idx), (y_key, y_idx), loc, xl, yl in zip(x_params, y_params, grid_locs, x_lims, y_lims):

    ax = fig.add_subplot(loc)
    for bf, c in zip(bfs, colors):
        try:
            a.plot_continuation(f'PAR({x_idx})', f'PAR({y_idx})', cont=f'{y_key}/{x_key}:{bf}', ax=ax,
                                line_color_stable=c, line_color_unstable=c)
        except KeyError:
            pass
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_xlim(xl)
    ax.set_ylim(yl)

# 1D continuations
##################

# settings
params = [("eta", 8, 3), ("eta", 8, 4), ("eta", 8, 5)]
dv = "U(1)"
grid_locs = [grid[2, 0], grid[2, 1], grid[2, 2]]
iv = "gs"
vals = a.additional_attributes[iv]

# plotting
for (param, idx, cont), loc, v in zip(params, grid_locs, vals):

    ax = fig.add_subplot(loc)
    try:
        a.plot_continuation(f'PAR({idx})', dv, cont=f'{param}:{cont}', ax=ax)
    except KeyError:
        pass
    ax.set_xlabel(param)
    ax.set_ylabel(dv)
    ax.set_title(f"{iv} = {v}")

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/rs_bifs.pdf')
plt.show()
