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
orig = pickle.load(open(f"results/pole_comp_eic_orig.p", "rb"))
new = pickle.load(open(f"results/pole_comp_eic_new.p", "rb"))

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
plt.rcParams['lines.linewidth'] = 3.0
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
l1 = a_orig.plot_continuation('PAR(36)', 'PAR(30)', cont='D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                              line_color_unstable='#5D6D7E', line_style_unstable='solid', alpha=0.5)
a_orig.plot_continuation('PAR(36)', 'PAR(30)', cont='D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                         line_color_unstable='#5D6D7E', line_style_unstable='solid', alpha=0.5)
l2 = a_orig.plot_continuation('PAR(36)', 'PAR(30)', cont='D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                              line_style_unstable='solid', alpha=0.5)
l3 = a_new.plot_continuation('PAR(36)', 'PAR(30)', cont='D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                             line_color_unstable='#5D6D7E', line_style_stable='dashed', linewidth=1.0)
a_new.plot_continuation('PAR(36)', 'PAR(30)', cont='D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                        line_color_unstable='#5D6D7E', line_style_stable='dashed', linewidth=1.0)
l4 = a_new.plot_continuation('PAR(36)', 'PAR(30)', cont='D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                             line_style_stable='dashed', linewidth=1.0)
plt.legend([l1, l2, l3, l4], ["Fold, eqs.(20-23)", "Hopf, eqs.(20-23)", "Fold, eqs.(24-27)", "Hopf, eqs.(24-27)"])
ax.set_xlabel(r'$I_{fs}$')
ax.set_ylabel(r'$\Delta_{fs}$')
ax.set_title('(A) 2D bifurcation diagram')
ax.set_xlim([10.0, 80.0])
ax.set_ylim([0.0, 1.6])

# plot the time signals\
for i, (pop, color) in enumerate(zip(["rs", "fs"], ["blue", "orange"])):

    # plot synaptic activation
    ax = fig.add_subplot(grid[i, 2:])
    ax.plot(orig['results'].index, orig['results'][pop], c=color)
    ax.plot(new['results'].index, new['results'][pop], c="white", linestyle="dashed", linewidth=1.0)
    if i == 0:
        plt.legend(['eqs.(20-23)', 'eqs.(24-27)'], loc=3).get_frame().set_facecolor("grey")
    elif i == 1:
        plt.legend(['eqs.(20-23)', 'eqs.(24-27)'], loc=2).get_frame().set_facecolor("grey")
        ax.set_xlabel('time (ms)')
    ax.set_ylabel(r'$r$ (Hz)')
    ax.set_yticks([0.0, 0.015, 0.03])
    ax.set_yticklabels(['0', '15', '30'])
    neuron = "Regular" if pop == "rs" else "Fast"
    ax.set_title(f"{neuron} spiking neurons")
    ax.set_xlim([np.min(orig['results'].index), np.max(orig['results'].index)])

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.01, hspace=0.05, wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/pole_comp_eic.pdf')
plt.show()
