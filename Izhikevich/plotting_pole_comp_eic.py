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
a_orig = ODESystem.from_file(f"results/eic.pkl", auto_dir=auto_dir)
a_new = ODESystem.from_file(f"results/eic2.pkl", auto_dir=auto_dir)
ds = a_orig.additional_attributes['deltas']
n = len(ds)

# load simulation data
orig_low = pickle.load(open(f"results/pole_comp_eic_orig.p", "rb"))
orig_high = pickle.load(open(f"results/pole_comp_eic2_orig.p", "rb"))
new_low = pickle.load(open(f"results/pole_comp_eic_new.p", "rb"))
new_high = pickle.load(open(f"results/pole_comp_eic2_new.p", "rb"))

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 4)
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
grid = gridspec.GridSpec(nrows=2, ncols=6, figure=fig)

# plot the 2D bifurcation diagram
n_points = 2
ax = fig.add_subplot(grid[:, :2])
a_orig.plot_continuation('PAR(30)', 'PAR(36)', cont='D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                         line_color_unstable='#5D6D7E', line_style_stable='dotted')
a_orig.plot_continuation('PAR(30)', 'PAR(36)', cont='D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                         line_color_unstable='#5D6D7E', line_style_stable='dotted')
a_orig.plot_continuation('PAR(30)', 'PAR(36)', cont='D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                         line_style_stable='dotted')
a_new.plot_continuation('PAR(30)', 'PAR(36)', cont='D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                        line_color_unstable='#5D6D7E', line_style_unstable='solid')
a_new.plot_continuation('PAR(30)', 'PAR(36)', cont='D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                        line_color_unstable='#5D6D7E', line_style_unstable='solid')
a_new.plot_continuation('PAR(30)', 'PAR(36)', cont='D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77')
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$\kappa$')
ax.set_title('(A) 2D bifurcation diagram')
# ax.set_ylim([10.0, 120.0])
# ax.set_xlim([20.0, 75.0])

# plot the time signals
data = [[orig_low, new_low], [orig_high, new_high]]
titles = [rf'(D) $\Delta_f = {ds[0]}$', rf'(E) $\Delta_f = {ds[1]}$']
for i, ((orig, new), title) in enumerate(zip(data, titles)):

    # plot synaptic activation
    ax = fig.add_subplot(grid[i, 2:])
    ax.plot(orig['results'].index, orig['results']['rs'])
    ax.plot(new['results'].index, new['results']['rs'])
    if i == 0:
        plt.legend(['eqs.(20-23)', 'eqs.(24-27)'], loc=2)
    ax.set_ylabel(r'$r$')
    ax.set_title(title)
    ax.set_xlim([np.min(orig['results'].index), np.max(orig['results'].index)])

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.01, hspace=0.05, wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/pole_comp_eic.pdf')
plt.show()
