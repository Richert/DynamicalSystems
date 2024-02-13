import matplotlib.pyplot as plt
from pycobi import ODESystem
import sys
import pickle
import numpy as np
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = ODESystem.from_file(f"results/etas_global.pkl", auto_dir=auto_dir)
kappas = a.additional_attributes['kappas']

# load simulation data
signals = {}
mf_models = ["mf_etas_global"]
snn_models = ["snn_etas", "snn_etas_global"]
models = mf_models + snn_models
for cond in ["no_sfa_1", "no_sfa_2", "weak_sfa_1", "weak_sfa_2", "strong_sfa_1", "strong_sfa_2"]:
    signals[cond] = {}
    for model in models:
        data = pickle.load(open(f"results/{model}_{cond}.pkl", "rb"))["results"]
        signals[cond][model] = data

# update bifurcation styles
a.update_bifurcation_style("HB", color="#76448A")

############
# plotting #
############

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# create figure layout
fig = plt.figure()
grid = fig.add_gridspec(nrows=4, ncols=3)

# choose colors
fold_color = "#5D6D7E"
hopf_color = "#148F77"
mf_colors = ["darkorange"]
snn_colors = ["black", "grey"]

# 2D continuations
##################

# no spike frequency adaptation
ax = fig.add_subplot(grid[:2, 0])
a.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:1:lp1', ax=ax, line_color_stable=fold_color,
                    line_color_unstable=fold_color, line_style_unstable="solid", bifurcation_legend=False)
a.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:1:lp2', ax=ax, line_color_stable=fold_color,
                    line_color_unstable=fold_color, line_style_unstable="solid", bifurcation_legend=False)
ax.set_title(rf'(A) Bifurcations for $\kappa = {kappas[0]}$ pA')
ax.set_ylabel(r'$b$ (nS)')
ax.set_xlabel(r'$I_{ext}$ (pA)')
ax.set_ylim([-25.0, 15.0])
ax.set_xlim([-200.0, 200.0])

# weak spike frequency adaptation
ax = fig.add_subplot(grid[:2, 1])
a.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:2:lp1', ax=ax, line_color_stable=fold_color,
                    line_color_unstable=fold_color, line_style_unstable="solid", bifurcation_legend=False)
a.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:2:lp2', ax=ax, line_color_stable=fold_color,
                    line_color_unstable=fold_color, line_style_unstable="solid", bifurcation_legend=False)
a.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:2:hb1', ax=ax, line_color_stable=hopf_color,
                    line_color_unstable=hopf_color, line_style_unstable="solid")
ax.set_title(rf'(B) Bifurcations for $\kappa = {kappas[1]}$ pA')
ax.set_ylabel(r'$b$ (nS)')
ax.set_xlabel(r'$I_{ext}$ (pA)')
ax.set_ylim([-25.0, 15.0])
ax.set_xlim([-200.0, 200.0])

# strong spike frequency adaptation
ax = fig.add_subplot(grid[:2, 2])
a.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:3:lp1', ax=ax, line_color_stable=fold_color,
                    line_color_unstable=fold_color, line_style_unstable="solid", bifurcation_legend=False)
a.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:3:lp2', ax=ax, line_color_stable=fold_color,
                    line_color_unstable=fold_color, line_style_unstable="solid", bifurcation_legend=False)
a.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:3:hb1', ax=ax, line_color_stable=hopf_color,
                    line_color_unstable=hopf_color, line_style_unstable="solid", bifurcation_legend=False)
a.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:3:hb2', ax=ax, line_color_stable=hopf_color,
                    line_color_unstable=hopf_color, line_style_unstable="solid", bifurcation_legend=False,
                    ignore=["UZ"])
ax.set_title(rf'(C) Bifurcations for $\kappa = {kappas[2]}$ pA')
ax.set_ylabel(r'$b$ (nS)')
ax.set_xlabel(r'$I_{ext}$ (pA)')
ax.set_ylim([-25.0, 15.0])
ax.set_xlim([-100.0, 300.0])

# Time series
#############

time = signals["no_sfa_1"]["mf_etas_global"].index

# no sfa, condition 1
cond = "no_sfa_1"
ax = fig.add_subplot(grid[2, 0])
for model, color, label in zip([snn_models[0], snn_models[1], mf_models[-1]],
                               [snn_colors[0], snn_colors[1], mf_colors[-1]],
                               ["SNN", "SNN (global)", "MF"]):
    data = signals[cond][model]["s"]
    ax.plot(time, data, color=color, label=label)
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$s$")
# ax.legend()
ax.set_title(f"(D) Phase Transition 1")

# no sfa, condition 2
cond = "no_sfa_2"
ax = fig.add_subplot(grid[3, 0])
for model, color, label in zip([snn_models[0], snn_models[1], mf_models[-1]],
                               [snn_colors[0], snn_colors[1], mf_colors[-1]],
                               ["SNN", "SNN (global)", "MF"]):
    data = signals[cond][model]["s"]
    ax.plot(time, data, color=color, label=label)
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$s$")
# ax.legend()
ax.set_title(f"(G) Phase Transition 2")

# weak sfa, condition 1
cond = "weak_sfa_1"
ax = fig.add_subplot(grid[2, 1])
for model, color, label in zip([snn_models[0], snn_models[1], mf_models[-1]],
                               [snn_colors[0], snn_colors[1], mf_colors[-1]],
                               ["SNN", "SNN (global)", "MF"]):
    data = signals[cond][model]["s"]
    ax.plot(time, data, color=color, label=label)
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$s$")
# ax.legend()
ax.set_title(f"(E) Phase Transition 1")

# weak sfa, condition 2
cond = "weak_sfa_2"
ax = fig.add_subplot(grid[3, 1])
for model, color, label in zip([snn_models[0], snn_models[1], mf_models[-1]],
                               [snn_colors[0], snn_colors[1], mf_colors[-1]],
                               ["SNN", "SNN (global)", "MF"]):
    data = signals[cond][model]["s"]
    ax.plot(time, data, color=color, label=label)
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$s$")
ax.legend()
ax.set_title(f"(H) Phase Transition 2")

# strong sfa, condition 1
cond = "strong_sfa_1"
ax = fig.add_subplot(grid[2, 2])
for model, color, label in zip([snn_models[0], snn_models[1], mf_models[-1]],
                               [snn_colors[0], snn_colors[1], mf_colors[-1]],
                               ["SNN", "SNN (global)", "MF"]):
    data = signals[cond][model]["s"]
    ax.plot(time, data, color=color, label=label)
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$s$")
# ax.legend()
ax.set_title(f"(F) Phase Transition 1")

# strong sfa, condition 2
cond = "strong_sfa_2"
ax = fig.add_subplot(grid[3, 2])
for model, color, label in zip([snn_models[0], snn_models[1], mf_models[-1]],
                               [snn_colors[0], snn_colors[1], mf_colors[-1]],
                               ["SNN", "SNN (global)", "MF"]):
    data = signals[cond][model]["s"]
    ax.plot(time, data, color=color, label=label)
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$s$")
# ax.legend()
ax.set_title(f"(I) Phase Transition 2")

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/etas.pdf')
plt.show()
