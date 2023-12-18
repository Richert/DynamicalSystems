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
a = ODESystem.from_file(f"results/etas.pkl", auto_dir=auto_dir)
ag = ODESystem.from_file(f"results/etas_global.pkl", auto_dir=auto_dir)
at = ODESystem.from_file(f"results/etas_test.pkl", auto_dir=auto_dir)
kappas = a.additional_attributes['kappas']

# load simulation data
signals = {}
for cond in ["no_sfa", "weak_sfa", "strong_sfa"]:
    signals[cond] = {"global": {}, "local": {}}
    for model in ["mf", "snn"]:
        global_data = pickle.load(open(f"results/{model}_etas_global_{cond}.pkl", "rb"))["results"]
        local_data = pickle.load(open(f"results/{model}_etas_{cond}.pkl", "rb"))["results"]
        signals[cond]["global"][model] = global_data
        signals[cond]["local"][model] = local_data

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

# 2D continuations
##################

# no spike frequency adaptation
ax = fig.add_subplot(grid[:2, 0])
for a_tmp, linestyle in zip([a, ag, at], ["solid", "dotted", "dashed"]):
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp1', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable=linestyle, line_style_stable=linestyle)
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp2', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable=linestyle, line_style_stable=linestyle)
ax.set_title(rf'(A) $\kappa = {kappas[0]}$ pA')
ax.set_ylabel(r'$\Delta$ (mv)')
ax.set_xlabel(r'$I_{ext}$ (pA)')

# weak spike frequency adaptation
ax = fig.add_subplot(grid[:2, 1])
for a_tmp, linestyle in zip([a, ag, at], ["solid", "dotted", "dashed"]):
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp3', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable=linestyle, line_style_stable=linestyle)
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:lp4', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable=linestyle, line_style_stable=linestyle)
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb1', ax=ax, line_color_stable='#148F77',
                            line_color_unstable='#148F77', line_style_unstable=linestyle, line_style_stable=linestyle)
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb2', ax=ax, line_color_stable='#148F77',
                            line_color_unstable='#148F77', line_style_unstable=linestyle, line_style_stable=linestyle)
ax.set_title(rf'(B) $\kappa = {kappas[1]}$ pA')
ax.set_ylabel(r'$\Delta$ (mv)')
ax.set_xlabel(r'$I_{ext}$ (pA)')
# ax.set_ylim([0.0, 4.0])
# ax.set_xlim([0.0, 80.0])

# strong spike frequency adaptation
ax = fig.add_subplot(grid[:2, 2])
for a_tmp, linestyle in zip([a, ag, at], ["solid", "dotted", "dashed"]):
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb3', ax=ax, line_color_stable='#148F77',
                            line_color_unstable='#148F77', line_style_unstable=linestyle, line_style_stable=linestyle)
    a_tmp.plot_continuation('PAR(8)', 'PAR(5)', cont=f'D/I:hb4', ax=ax, line_color_stable='#148F77',
                            line_color_unstable='#148F77', line_style_unstable=linestyle, line_style_stable=linestyle)
ax.set_ylabel(r'$\Delta$ (mv)')
ax.set_xlabel(r'$I_{ext}$ (pA)')
ax.set_title(rf'(C) $\kappa = {kappas[2]}$ pA')
# ax.set_ylim([0.0, 2.0])
# ax.set_xlim([0.0, 80.0])

# Time series
#############

# no spike frequency adaptation
cond = "no_sfa"
approximation = ["local", "global"]
rows = [2, 3]
titles = ["(D) Neuron-specific recovery variables", "(G) Global recovery variable"]
for approx, row, title in zip(approximation, rows, titles):
    mf_data = signals[cond][approx]["mf"]
    snn_data = signals[cond][approx]["snn"]
    ax = fig.add_subplot(grid[row, 0])
    ax.plot(mf_data.index, np.mean(snn_data, axis=1), label="spiking network")
    ax.plot(mf_data.index, mf_data["s"], label="mean-field")
    ax.legend()
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$s$")
    ax.set_ylim([0.0, 0.2])
    ax.set_title(title)

# weak spike frequency adaptation
cond = "weak_sfa"
approximation = ["local", "global"]
rows = [2, 3]
titles = ["(E)", "(H)"]
for approx, row, title in zip(approximation, rows, titles):
    mf_data = signals[cond][approx]["mf"]
    snn_data = signals[cond][approx]["snn"]
    ax = fig.add_subplot(grid[row, 1])
    ax.plot(mf_data.index, np.mean(snn_data, axis=1), label="spiking network")
    ax.plot(mf_data.index, mf_data["s"], label="mean-field")
    ax.legend()
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$s$")
    ax.set_ylim([0.0, 0.2])
    ax.set_title(title)

# strong spike frequency adaptation
cond = "strong_sfa"
approximation = ["local", "global"]
rows = [2, 3]
titles = ["(F)", "(I)"]
for approx, row, title in zip(approximation, rows, titles):
    mf_data = signals[cond][approx]["mf"]
    snn_data = signals[cond][approx]["snn"]
    ax = fig.add_subplot(grid[row, 2])
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
plt.savefig(f'results/etas.pdf')
plt.show()
