from matplotlib import gridspec
import matplotlib.pyplot as plt
from pycobi import ODESystem
import sys
import pickle
import numpy as np
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = ODESystem.from_file(f"results/bs.pkl", auto_dir=auto_dir)
ag = ODESystem.from_file(f"results/bs_global.pkl", auto_dir=auto_dir)
deltas = a.additional_attributes['deltas']
kappas = a.additional_attributes['kappas']

# load simulation data
signals = {}
for key in ["mf_etas_global", "mf_etas", "snn_etas_global", "snn_etas",
            "mf_etas_global2", "mf_etas2", "snn_etas_global2", "snn_etas2"
            ]:
    signals[key] = pickle.load(open(f"results/{key}.pkl", "rb"))["results"]

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
grid = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)

# 2D continuations
##################

# low kappa
ax = fig.add_subplot(grid[:2, 0])
for a_tmp, linestyle in zip([a, ag], ["solid", "dotted"]):
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp1', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable=linestyle, line_style_stable=linestyle)
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp2', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable=linestyle, line_style_stable=linestyle)
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb1', ax=ax, line_color_stable='#148F77',
                            line_color_unstable='#148F77', line_style_unstable=linestyle, line_style_stable=linestyle)
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb2', ax=ax, line_color_stable='#148F77',
                            line_color_unstable='#148F77', line_style_unstable=linestyle, line_style_stable=linestyle)
ax.set_title(rf'(A) $\kappa = {kappas[0]}$ pA')
ax.set_ylabel(r'$\Delta$ (mv)')
ax.set_xlabel(r'$I_{ext}$ (pA)')
# ax.set_ylim([0.0, 4.0])
# ax.set_xlim([0.0, 80.0])

# high kappa
ax = fig.add_subplot(grid[:2, 1])
for a_tmp, linestyle in zip([a, ag], ["solid", "dotted"]):
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb3', ax=ax, line_color_stable='#148F77',
                            line_color_unstable='#148F77', line_style_unstable=linestyle, line_style_stable=linestyle)
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb4', ax=ax, line_color_stable='#148F77',
                            line_color_unstable='#148F77', line_style_unstable=linestyle, line_style_stable=linestyle)
ax.set_ylabel(r'$\Delta$ (mv)')
ax.set_xlabel(r'$I_{ext}$ (pA)')
ax.set_title(rf'(B) $\kappa = {kappas[1]}$ pA')
# ax.set_ylim([0.0, 2.0])
# ax.set_xlim([0.0, 80.0])

# Time series
#############

# for small kappa
conditions = ["etas", "etas_global"]
rows = [2, 3]
titles = ["(C) Neuron-specific recovery variables", "(D) Global recovery variable"]
for cond, row, title in zip(conditions, rows, titles):
    mf_data = signals[f"mf_{cond}"]
    snn_data = signals[f"snn_{cond}"]
    ax = fig.add_subplot(grid[row, 0])
    ax.plot(mf_data.index, np.mean(snn_data, axis=1), label="spiking network")
    ax.plot(mf_data.index, mf_data["s"], label="mean-field")
    ax.legend()
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$s$")
    ax.set_ylim([0.0, 0.2])
    ax.set_title(title)

# for large kappa
conditions = ["etas", "etas_global"]
rows = [2, 3]
titles = ["(C) Neuron-specific recovery variables", "(D) Global recovery variable"]
for cond, row, title in zip(conditions, rows, titles):
    mf_data = signals[f"mf_{cond}2"]
    snn_data = signals[f"snn_{cond}2"]
    ax = fig.add_subplot(grid[row, 1])
    ax.plot(mf_data.index, np.mean(snn_data, axis=1), label="spiking network")
    ax.plot(mf_data.index, mf_data["s"], label="mean-field")
    ax.legend()
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$s$")
    ax.set_ylim([0.0, 0.2])
    ax.set_title(title)

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/bs.pdf')
plt.show()
