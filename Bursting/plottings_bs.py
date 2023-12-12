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
a = ODESystem.from_file(f"results/bs.pkl", auto_dir=auto_dir)
deltas = a.additional_attributes['deltas']
kappas = a.additional_attributes['kappas']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (8, 4)
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
grid = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)

# 2D continuations
##################

# low kappa
ax = fig.add_subplot(grid[0, 0])
a.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
a.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb2', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta$ (mv)')
ax.set_xlabel(r'$I_{ext}$ (pA)')
ax.set_title(rf'(A) $\kappa = {kappas[0]}$ pA')
# ax.set_ylim([0.0, 4.0])
# ax.set_xlim([0.0, 80.0])

# high kappa
ax = fig.add_subplot(grid[0, 1])
a.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb3', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
a.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb4', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
ax.set_ylabel(r'$\Delta$ (mv)')
ax.set_xlabel(r'$I_{ext}$ (pA)')
ax.set_title(rf'(A) $\kappa = {kappas[1]}$ pA')
# ax.set_ylim([0.0, 2.0])
# ax.set_xlim([0.0, 80.0])

# Time series
#############

# conditions = ["hom_low_sfa", "het_low_sfa", "hom_high_sfa", "het_high_sfa"]
# titles = [r"(B) $\kappa_{rs} = 10.0$ pA, $\Delta_{rs} = 0.5$", r"(C) $\kappa_{rs} = 10.0$ pA, $\Delta_{rs} = 2.0$",
#           r"(E) $\kappa_{rs} = 100.0$ pA, $\Delta_{rs} = 0.5$", r"(F) $\kappa_{rs} = 100.0$ pA, $\Delta_{rs} = 1.5$"]
# subplots = [0, 1, 2, 3]
# for cond, title, idx in zip(conditions, titles, subplots):
#     data = pickle.load(open(f"results/rs_{cond}.p", "rb"))
#     ax = fig.add_subplot(grid[idx, 1:])
#     ax.plot(data["mf"].index, data["mf"]["s"], label="mean-field")
#     ax.plot(data["mf"].index, np.mean(data["snn"], axis=1), label="spiking network")
#     if idx == 0:
#         plt.legend()
#     ax.set_xlabel("time (ms)")
#     ax.set_ylabel(r"$s$")
#     ax.set_title(title)

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/bs.pdf')
plt.show()
