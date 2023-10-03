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
a_eic = ODESystem.from_file(f"results/eic_shadowing.pkl", auto_dir=auto_dir)
a_rs = ODESystem.from_file(f"results/rs.pkl", auto_dir=auto_dir)

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

############
# plotting #
############

# create figure layout
fig = plt.figure(figsize=(12, 8), constrained_layout=True)
grid_highlvl = fig.add_gridspec(1, 2)

# time series
#############

grid = grid_highlvl[1].subgridspec(6, 1)

# RS conditions
conditions = ["rs_low_sfa", "rs_high_sfa"]
titles = [r"(C) RS: $\Delta_{rs} = 0.5$ mV, $\kappa_{rs} = 10.0$",
          r"(D) RS: $\Delta_{rs} = 0.5$ mV, $\kappa_{rs} = 100.0$",
          ]
subplots = [0, 1]
ticks = [0.2, 0.1]
tau = 6.0
ax_init = []
for cond, title, idx, ytick in zip(conditions, titles, subplots, ticks):
    data = pickle.load(open(f"results/{cond}.p", "rb"))
    if idx == 0:
        ax = fig.add_subplot(grid[idx])
        ax_init.append(ax)
    else:
        ax = fig.add_subplot(grid[idx], sharex=ax_init[0])
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

# EIC conditions
conditions = ["eic_het_low_sfa", "eic_het_high_sfa", "eic_hom_low_sfa", "eic_hom_high_sfa"]
titles = [
    r"(G) RS-FS: $\Delta_{fs} = 2.0$ mV, $\kappa_{rs} = 10.0$",
    r"(H) RS-FS: $\Delta_{fs} = 2.0$ mV, $\kappa_{rs} = 100.0$",
    r"(K) RS-FS: $\Delta_{fs} = 0.2$ mV, $\kappa_{rs} = 10.0$",
    r"(L) RS-FS: $\Delta_{fs} = 0.2$ mV, $\kappa_{rs} = 100.0$"
          ]
subplots = [2, 3, 4, 5]
ticks = [25, 15, 40, 30]
for cond, title, idx, ytick in zip(conditions, titles, subplots, ticks):
    data = pickle.load(open(f"results/{cond}.p", "rb"))["results"]
    snn_data = pickle.load(open(f"results/snn_{cond}.p", "rb"))["rs_results"]
    ax = fig.add_subplot(grid[idx], sharex=ax_init[0])
    rs, fs = data["rs"], data["fs"]
    ax.plot(rs.index, snn_data[5000:]/6.0, label="spiking network", color="black")
    ax.plot(rs, label="mean-field", color="darkorange")
    # ax.plot(fs, label="FS", color="darkorange")
    ax.set_yticks([0.0, ytick*1e-3])
    ax.set_yticklabels(['0', str(ytick)])
    if idx == 2:
        ax.legend()
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$r$ (Hz)")
    ax.set_title(title)

# 2D continuations
##################

grid = grid_highlvl[0].subgridspec(3, 2)

# RS: d_rs = 10.0
ax = fig.add_subplot(grid[0, 0])
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp1', ax=ax, line_color_stable='#5D6D7E',
                       line_color_unstable='#5D6D7E', line_style_unstable='solid')
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp2', ax=ax, line_color_stable='#5D6D7E',
                       line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{rs}$ (pA)')
ax.set_title(r'(A) RS: $\kappa_{rs} = 10.0$ pA')
ax.set_ylim([0.0, 4.0])
ax.set_xlim([10.0, 70.0])
ax.axhline(y=0.5, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=30.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=50.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)

# RS: d_rs = 100.0
ax = fig.add_subplot(grid[0, 1])
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb1', ax=ax, line_color_stable='#148F77',
                       line_color_unstable='#148F77', line_style_unstable='solid')
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp3', ax=ax, line_color_stable='#5D6D7E',
                       line_color_unstable='#5D6D7E', line_style_unstable='solid')
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp4', ax=ax, line_color_stable='#5D6D7E',
                       line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{rs}$ (pA)')
ax.set_title(r'(B) RS: $\kappa_{rs} = 100.0$ pA')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([10.0, 70.0])
ax.axhline(y=0.5, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=32.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=52.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)

# EIC: Delta_fs = 1.0, d_rs = 10.0
ax = fig.add_subplot(grid[1, 0])
a_eic.plot_continuation('PAR(30)', 'PAR(6)', cont=f'D_rs/I_fs:3:lp1', ax=ax, line_color_stable='#5D6D7E',
                        line_color_unstable='#5D6D7E', line_style_unstable='solid')
a_eic.plot_continuation('PAR(30)', 'PAR(6)', cont=f'D_rs/I_fs:3:lp2', ax=ax, line_color_stable='#5D6D7E',
                        line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{fs}$ (pA)')
ax.set_title(r'(E) RS-FS: $\Delta_{fs} = 2.0$ mV, $\kappa_{rs} = 10.0$')
ax.set_ylim([0.0, 2.0])
ax.set_xlim([70.0, 10.0])
ax.axhline(y=0.5, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=45.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=25.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)

# Delta_fs = 1.0, d_rs = 100.0
ax = fig.add_subplot(grid[1, 1])
a_eic.plot_continuation('PAR(30)', 'PAR(6)', cont=f'D_rs/I_fs:4:hb1', ax=ax, line_color_stable='#148F77',
                        line_color_unstable='#148F77', line_style_unstable='solid')
a_eic.plot_continuation('PAR(30)', 'PAR(6)', cont=f'D_rs/I_fs:4:lp1', ax=ax, line_color_stable='#5D6D7E',
                        line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{fs}$ (pA)')
ax.set_title(r'(F) RS-FS: $\Delta_{fs} = 2.0$ mV, $\kappa_{rs} = 100.0$')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([70.0, 10.0])
ax.axhline(y=0.5, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=45.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=25.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)

# EIC: Delta_fs = 0.1, d_rs = 10.0
ax = fig.add_subplot(grid[2, 0])
a_eic.plot_continuation('PAR(30)', 'PAR(6)', cont=f'D_rs/I_fs:1:hb1', ax=ax, line_color_stable='#148F77',
                        line_color_unstable='#148F77', line_style_unstable='solid')
# a_eic.plot_continuation('PAR(30)', 'PAR(6)', cont=f'D_rs/I_fs:1:hb2', ax=ax, line_color_stable='#148F77',
#                         line_color_unstable='#148F77', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{fs}$ (pA)')
ax.set_title(r'(I) RS-FS: $\Delta_{fs} = 0.2$ mV, $\kappa_{rs} = 10.0$')
ax.set_ylim([0.0, 2.0])
ax.set_xlim([70.0, 10.0])
ax.axhline(y=0.5, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=45.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=25.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)

# EIC: Delta_fs = 0.1, d_rs = 100.0
ax = fig.add_subplot(grid[2, 1])
a_eic.plot_continuation('PAR(30)', 'PAR(6)', cont=f'D_rs/I_fs:2:hb1', ax=ax, line_color_stable='#148F77',
                        line_color_unstable='#148F77', line_style_unstable='solid')
a_eic.plot_continuation('PAR(30)', 'PAR(6)', cont=f'D_rs/I_fs:2:hb2', ax=ax, line_color_stable='#148F77',
                        line_color_unstable='#148F77', line_style_unstable='solid')
a_eic.plot_continuation('PAR(30)', 'PAR(6)', cont=f'D_rs/I_fs:2:lp1', ax=ax, line_color_stable='#5D6D7E',
                        line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{fs}$ (pA)')
ax.set_title(r'(J) RS-FS: $\Delta_{fs} = 0.2$ mV, $\kappa_{rs} = 100.0$')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([70.0, 10.0])
ax.axhline(y=0.5, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=45.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=25.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/eic_shadowing.svg')
plt.show()
