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
a = ODESystem.from_file(f"results/eic_shadowing.pkl", auto_dir=auto_dir)
deltas = a.additional_attributes['deltas']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 8)
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
grid = gridspec.GridSpec(nrows=6, ncols=4, figure=fig)

# 2D continuations
##################

# Delta_fs = 0.1, d_rs = 10.0
ax = fig.add_subplot(grid[:2, 0])
a.plot_continuation('PAR(26)', 'PAR(6)', cont=f'D_rs/I_fs:1:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
# a.plot_continuation('PAR(30)', 'PAR(6)', cont=f'D_rs/I_fs:1:hb2', ax=ax, line_color_stable='#148F77',
#                     line_color_unstable='#148F77', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{fs}$ (pA)')
ax.set_title(r'(A) $\Delta_{fs} = 0.2$ mV, $\kappa_{rs} = 10.0$')
# ax.set_ylim([0.0, 1.7])
# ax.set_xlim([10.0, 70.0])

# Delta_fs = 0.1, d_rs = 100.0
ax = fig.add_subplot(grid[2:4, 0])
a.plot_continuation('PAR(26)', 'PAR(6)', cont=f'D_rs/I_fs:2:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
a.plot_continuation('PAR(26)', 'PAR(6)', cont=f'D_rs/I_fs:2:hb2', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
a.plot_continuation('PAR(26)', 'PAR(6)', cont=f'D_rs/I_fs:2:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{fs}$ (pA)')
ax.set_title(r'(B) $\Delta_{fs} = 0.2$ mV, $\kappa_{rs} = 100.0$')
# ax.set_ylim([0.0, 1.7])
# ax.set_xlim([10.0, 70.0])

# Delta_fs = 1.0, d_rs = 10.0
ax = fig.add_subplot(grid[:2, 1])
a.plot_continuation('PAR(26)', 'PAR(6)', cont=f'D_rs/I_fs:3:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(26)', 'PAR(6)', cont=f'D_rs/I_fs:3:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{fs}$ (pA)')
ax.set_title(r'(E) $\Delta_{fs} = 2.0$ mV, $\kappa_{rs} = 10.0$')
# ax.set_ylim([0.0, 1.7])
# ax.set_xlim([10.0, 70.0])

# Delta_fs = 1.0, d_rs = 100.0
ax = fig.add_subplot(grid[2:4, 1])
a.plot_continuation('PAR(26)', 'PAR(6)', cont=f'D_rs/I_fs:4:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
a.plot_continuation('PAR(26)', 'PAR(6)', cont=f'D_rs/I_fs:4:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{fs}$ (pA)')
ax.set_title(r'(F) $\Delta_{fs} = 2.0$ mV, $\kappa_{rs} = 100.0$')
# ax.set_ylim([0.0, 1.7])
# ax.set_xlim([10.0, 70.0])

# Delta_rs = 0.1, d_rs = 10.0
ax = fig.add_subplot(grid[4:, 0])
a.plot_continuation('PAR(15)', 'PAR(24)', cont=f'D_fs/I_rs:1:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
a.plot_continuation('PAR(15)', 'PAR(24)', cont=f'D_fs/I_rs:1:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(15)', 'PAR(24)', cont=f'D_fs/I_rs:1:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{fs}$ (mv)')
ax.set_xlabel(r'$I_{rs}$ (pA)')
ax.set_title(r'(G) $\Delta_{rs} = 0.1$ mV, $\kappa_{rs} = 10.0$')
# ax.set_ylim([0.0, 1.7])
# ax.set_xlim([10.0, 70.0])

# time series
#############

conditions = ["hom_low_sfa", "het_low_sfa", "hom_high_sfa", "het_high_sfa"]
titles = [r"(C) $\Delta_{fs} = 0.2$ mV, $\kappa_{rs} = 10.0$", r"(D) $\Delta_{fs} = 2.0$ mV, $\kappa_{rs} = 10.0$",
          r"(G) $\Delta_{fs} = 0.2$ mV, $\kappa_{rs} = 100.0$", r"(H) $\Delta_{fs} = 2.0$ mV, $\kappa_{rs} = 100.0$"]
subplots = [0, 1, 2, 3]
for cond, title, idx in zip(conditions, titles, subplots):
    data = pickle.load(open(f"results/eic_{cond}.p", "rb"))["results"]
    ax = fig.add_subplot(grid[idx, 2:])
    ax.plot(data.index, data["rs"], label="RS")
    ax.plot(data.index, data["fs"], label="FS")
    if idx == 0:
        plt.legend()
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$r$ (Hz)")
    ax.set_title(title)

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/eic_shadowing.pdf')
plt.show()
