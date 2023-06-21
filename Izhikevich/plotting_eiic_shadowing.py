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
grid = gridspec.GridSpec(nrows=2, ncols=4, figure=fig)

# 2D continuations
##################

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

# Delta_lts = 1.8, d_rs = 10.0
ax = fig.add_subplot(grid[0, 1])
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:3:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:3:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_title(r'(B) $\Delta_{lts} = 1.8$ mV, $\kappa_{rs} = 10.0$')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([160.0, 80.0])

# Delta_lts = 0.6, d_rs = 10.0
ax = fig.add_subplot(grid[0, 2])
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:2:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:2:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:2:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_title(r'(C) $\Delta_{lts} = 0.6$ mV, $\kappa_{rs} = 10.0$')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([160.0, 80.0])

# Delta_lts = 0.1, d_rs = 10.0
ax = fig.add_subplot(grid[0, 3])
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
ax.set_title(r'(D) $\Delta_{lts} = 0.1$ mV, $\kappa_{rs} = 10.0$')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([160.0, 80.0])

# RS: d_rs = 100.0
ax = fig.add_subplot(grid[1, 0])
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb1', ax=ax, line_color_stable='#148F77',
                       line_color_unstable='#148F77', line_style_unstable='solid')
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp3', ax=ax, line_color_stable='#5D6D7E',
                       line_color_unstable='#5D6D7E', line_style_unstable='solid')
a_rs.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp4', ax=ax, line_color_stable='#5D6D7E',
                       line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{rs}$ (pA)')
ax.set_title(r'(E) RS: $\kappa_{rs} = 100.0$ pA')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([40.0, 70.0])

# Delta_lts = 1.8, d_rs = 100.0
ax = fig.add_subplot(grid[1, 1])
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:6:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:6:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:6:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_title(r'(F) $\Delta_{lts} = 1.8$ mV, $\kappa_{rs} = 100.0$')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([120.0, 60.0])

# Delta_lts = 0.6, d_rs = 100.0
ax = fig.add_subplot(grid[1, 2])
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:5:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:5:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:5:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_title(r'(G) $\Delta_{lts} = 0.6$ mV, $\kappa_{rs} = 100.0$')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([120.0, 60.0])

# Delta_lts = 0.1, d_rs = 100.0
ax = fig.add_subplot(grid[1, 3])
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:4:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:4:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(45)', 'PAR(6)', cont=f'D_rs/I_lts:4:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta_{rs}$ (mv)')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_title(r'(H) $\Delta_{lts} = 0.1$ mV, $\kappa_{rs} = 100.0$')
ax.set_ylim([0.0, 1.7])
ax.set_xlim([120.0, 60.0])

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/eiic_shadowing.svg')
plt.show()
