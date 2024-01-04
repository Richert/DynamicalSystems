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
a = ODESystem.from_file(f"results/etas_sfa.pkl", auto_dir=auto_dir)
ag = ODESystem.from_file(f"results/etas_sfa_global.pkl", auto_dir=auto_dir)
kappas = a.additional_attributes['deltas']

# load simulation data
signals = {}
mf_models = ["mf_etas_sfa", "mf_etas_corrected"]
snn_models = ["snn_etas", "snn_etas_sfa"]
models = mf_models + snn_models
for cond in ["low_sfa", "med_sfa", "high_sfa"]:
    signals[cond] = {}
    for model in models:
        data = pickle.load(open(f"results/{model}_{cond}.pkl", "rb"))["results"]
        signals[cond][model] = data

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
grid = gridspec.GridSpec(nrows=4, ncols=3, figure=fig)

# choose colors
mf_colors = ["#5D6D7E", "#1d9ce0", "#e06b1d"]
snn_colors = ["#000000", "#000000"]

# 2D continuations
##################

# no spike frequency adaptation
ax = fig.add_subplot(grid[:2, 0])
lines = []
for a_tmp, color in zip([a, ag], mf_colors):
    line = a_tmp.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:lp1', ax=ax, line_color_stable=color,
                                   line_color_unstable=color, line_style_unstable="solid")
    a_tmp.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:lp2', ax=ax, line_color_stable=color,
                            line_color_unstable=color, line_style_unstable="solid")
    a_tmp.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:hb1', ax=ax, line_color_stable=color,
                            line_color_unstable=color, line_style_unstable="dotted", line_style_stable="dotted")
    a_tmp.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:hb2', ax=ax, line_color_stable=color,
                            line_color_unstable=color, line_style_unstable="dotted", line_style_stable="dotted")
    lines.append(line)
ax.set_title(rf'(A) Bifurcations for $\kappa = {kappas[0]}$ pA')
ax.set_ylabel(r'$b$ (nS)')
ax.set_xlabel(r'$I_{ext}$ (pA)')
# ax.legend(lines, ["MF", "MF (global)", "MF (corrected)"])
ax.set_ylim([-20.0, 15.0])
ax.set_xlim([-200.0, 200.0])

# weak spike frequency adaptation
ax = fig.add_subplot(grid[:2, 1])
lines = []
for a_tmp, color in zip([a, ag], mf_colors):
    line = a_tmp.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:lp3', ax=ax, line_color_stable=color,
                                   line_color_unstable=color, line_style_unstable="solid")
    a_tmp.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:lp4', ax=ax, line_color_stable=color,
                            line_color_unstable=color, line_style_unstable="solid")
    a_tmp.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:hb3', ax=ax, line_color_stable=color,
                            line_color_unstable=color, line_style_unstable="dotted", line_style_stable="dotted")
    a_tmp.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:hb4', ax=ax, line_color_stable=color,
                            line_color_unstable=color, line_style_unstable="dotted", line_style_stable="dotted")
    lines.append(line)
ax.set_title(rf'(B) Bifurcations for $\kappa = {kappas[1]}$ pA')
ax.set_ylabel(r'$b$ (nS)')
ax.set_xlabel(r'$I_{ext}$ (pA)')
# ax.legend(lines, ["MF", "MF (global)", "MF (corrected)"])
ax.set_ylim([-20.0, 15.0])
ax.set_xlim([-200.0, 200.0])

# strong spike frequency adaptation
ax = fig.add_subplot(grid[:2, 2])
lines = []
for a_tmp, color in zip([a, ag], mf_colors):
    line = a_tmp.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:hb5', ax=ax, line_color_stable=color,
                                   line_color_unstable=color, line_style_unstable="dotted", line_style_stable="dotted")
    a_tmp.plot_continuation('PAR(8)', 'PAR(9)', cont=f'b/I:hb6', ax=ax, line_color_stable=color,
                            line_color_unstable=color, line_style_unstable="dotted", line_style_stable="dotted")
    lines.append(line)
ax.set_title(rf'(C) Bifurcations for $\kappa = {kappas[2]}$ pA')
ax.set_ylabel(r'$b$ (nS)')
ax.set_xlabel(r'$I_{ext}$ (pA)')
# ax.legend(lines, ["MF", "MF (global)", "MF (corrected)"])
ax.set_ylim([-20.0, 15.0])
ax.set_xlim([-200.0, 200.0])

# Time series
#############

time = signals["low_sfa"]["mf_etas_sfa"].index

# low heterogeneity, comparison against SNN
cond = "low_sfa"
ax = fig.add_subplot(grid[2, 0])
for model, color, label in zip([snn_models[0], mf_models[0], mf_models[1]],
                               [snn_colors[0], mf_colors[0], mf_colors[1]],
                               ["SNN", "MF", "MF (global)"]):
    data = signals[cond][model]
    data = np.mean(data, axis=1) if "SNN" in label else data["s"]
    ax.plot(time, data, color=color, label=label)
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$s$")
# ax.legend()
ax.set_title(f"(D) Comparison against SNN")

# low heterogeneity, comparison against SNN (global)
ax = fig.add_subplot(grid[3, 0])
for model, color, label in zip([snn_models[1], mf_models[0], mf_models[1]],
                               [snn_colors[1], mf_colors[0], mf_colors[1]],
                               ["SNN (global)", "MF", "MF (global)"]):
    data = signals[cond][model]
    data = np.mean(data, axis=1) if "SNN" in label else data["s"]
    ax.plot(time, data, color=color, label=label)
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$s$")
# ax.legend()
ax.set_title(f"(G) Comparison against SNN (global)")

# medium heterogeneity, comparison against SNN
cond = "med_sfa"
ax = fig.add_subplot(grid[2, 1])
for model, color, label in zip([snn_models[0], mf_models[0], mf_models[1]],
                               [snn_colors[0], mf_colors[0], mf_colors[1]],
                               ["SNN", "MF", "MF (global)"]):
    data = signals[cond][model]
    data = np.mean(data, axis=1) if "SNN" in label else data["s"]
    ax.plot(time, data, color=color, label=label)
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$s$")
# ax.legend()
ax.set_title(f"(E) Comparison against SNN")

# medium heterogeneity, comparison against SNN (global)
ax = fig.add_subplot(grid[3, 1])
for model, color, label in zip([snn_models[1], mf_models[0], mf_models[1]],
                               [snn_colors[1], mf_colors[0], mf_colors[1]],
                               ["SNN (global)", "MF", "MF (global)"]):
    data = signals[cond][model]
    data = np.mean(data, axis=1) if "SNN" in label else data["s"]
    ax.plot(time, data, color=color, label=label)
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$s$")
# ax.legend()
ax.set_title(f"(H) Comparison against SNN (global)")

# high heterogeneity
cond = "high_sfa"
ax = fig.add_subplot(grid[2, 2])
for model, color, label in zip([snn_models[0], mf_models[0], mf_models[1]],
                               [snn_colors[0], mf_colors[0], mf_colors[1]],
                               ["SNN", "MF", "MF (global)"]):
    data = signals[cond][model]
    data = np.mean(data, axis=1) if "SNN" in label else data["s"]
    ax.plot(time, data, color=color, label=label)
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$s$")
ax.legend()
ax.set_title(f"(F) Comparison against SNN")

# high heterogeneity, comparison against SNN (global)
ax = fig.add_subplot(grid[3, 2])
for model, color, label in zip([snn_models[1], mf_models[0], mf_models[1]],
                               [snn_colors[1], mf_colors[0], mf_colors[1]],
                               ["SNN (global)", "MF", "MF (global)"]):
    data = signals[cond][model]
    data = np.mean(data, axis=1) if "SNN" in label else data["s"]
    ax.plot(time, data, color=color, label=label)
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$s$")
ax.legend()
ax.set_title(f"(I) Comparison against SNN (global)")

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/etas_sfa.pdf')
plt.show()
