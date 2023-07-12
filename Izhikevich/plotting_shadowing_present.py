from matplotlib import gridspec
import matplotlib.pyplot as plt
from pycobi import ODESystem
import sys
import numpy as np
import pickle
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = ODESystem.from_file(f"results/eiic_shadowing.pkl", auto_dir=auto_dir)
deltas = a.additional_attributes['deltas']
a_rs = ODESystem.from_file(f"results/rs.pkl", auto_dir=auto_dir)

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

a.update_bifurcation_style("HB", color="#76448A")

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=4, ncols=3, figure=fig)

# timeseries
############

# RS conditions
conditions = ["rs_low_sfa", "rs_high_sfa"]
titles = [r"One-population model (RS)",
          r"",
          ]
subplots = [0, 1]
ticks = [0.2, 0.1]
tau = 6.0
ax_init = []
for cond, title, idx, ytick in zip(conditions, titles, subplots, ticks):
    data = pickle.load(open(f"results/{cond}.p", "rb"))
    if idx == 0:
        ax = fig.add_subplot(grid[idx, 1:])
        ax_init.append(ax)
    else:
        ax = fig.add_subplot(grid[idx, 1:], sharex=ax_init[0])
    mf, snn = data["mf"], data["snn"]
    ax.plot(mf.index, np.mean(snn, axis=1), label="spiking neurons", color="black")
    ax.plot(mf.index, mf["s"], label="mean-field", color="royalblue")
    if idx == 0:
        ax.legend()
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$r$ (Hz)")
    ax.set_yticks([0.0, ytick])
    ax.set_yticklabels(['0', str(int(ytick*1e3/tau))])
    ax.set_title(title)

# 2D continuations
##################

# RS: d_rs = 10.0
ax = fig.add_subplot(grid[:2, 0])
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp1', ax=ax, line_color_stable='#5D6D7E',
                       line_color_unstable='#5D6D7E', line_style_unstable='solid')
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp2', ax=ax, line_color_stable='#5D6D7E',
                       line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{rs}$ (pA)')
ax.set_title(r'Three-population model (RS, FS, LTS)')
ax.set_ylim([0.0, 4.0])
ax.set_xlim([10.0, 70.0])

# Delta_lts = 1.8, d_rs = 10.0
ax = fig.add_subplot(grid[2:, 0])
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:3:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:3:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_title(r'$\Delta_{lts} = 1.8$ mV')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([160.0, 80.0])

# Delta_lts = 0.6, d_rs = 10.0
ax = fig.add_subplot(grid[2:, 1])
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:2:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:2:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:2:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_title(r'$\Delta_{lts} = 0.6$ mV')
ax.legend(["Quiescent", "Persistent Spiking", "Oscillatory", "Bistable"])
ax.set_ylim([0.0, 1.7])
ax.set_xlim([160.0, 80.0])

# Delta_lts = 0.1, d_rs = 10.0
ax = fig.add_subplot(grid[2:, 2])
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:1:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:1:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:1:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:1:hb2', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_title(r'$\Delta_{lts} = 0.1$ mV')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([160.0, 80.0])

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/shadowing_present.svg')
plt.show()
