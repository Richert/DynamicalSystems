import pickle
from pycobi import ODESystem
import matplotlib.pyplot as plt

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.2
markersize = 6

# load data
ode = ODESystem.from_file("results/mg_bifurcations.pkl", auto_dir="~/PycharmProjects/auto-07p")
neuron = pickle.load(open("results/spn_singleneuron.pkl", "rb"))
snn_intact = pickle.load(open("results/snn_mg_intact.pkl", "rb"))
snn_ablated = pickle.load(open("results/snn_mg_ablated.pkl", "rb"))
mf_intact = pickle.load(open("results/mf_mg_intact.pkl", "rb"))["results"]
mf_ablated = pickle.load(open("results/mf_mg_ablated.pkl", "rb"))["results"]

# figure layout
fig = plt.figure(figsize=(8, 4))
grid = fig.add_gridspec(nrows=3, ncols=3)

# plot single neuron dynamics
ax = fig.add_subplot(grid[0, :])
ax.plot(neuron["v"])
ax.set_ylabel(r"$v$ (mV)")
ax.set_xlabel("time (ms)")
ax.set_title("(A) MSN neuron dynamics")

# plot 2D bifurcation diagram
ax = fig.add_subplot(grid[1:, 0])
gs = [1.0, 2.0, 4.0, 8.0]
lines = []
colors = ["black", "red", "blue", "green"]
for i in range(len(gs)):
    l = ode.plot_continuation("p/mg_op/s_e", "p/mg_op/alpha", cont=f"alpha/s_e:{i+1}", ax=ax,
                              line_style_unstable="solid", line_color_stable=colors[i], line_color_unstable=colors[i])
    lines.append(l)
ax.set_xlabel(r"$s_e$")
ax.set_ylabel(r"$\alpha$")
ax.legend(lines, [fr"$g_i = {g}$ nS" for g in gs])
ax.set_title("(B) 2D bifurcation diagram")
ax.set_yticks([0.0, 0.0002, 0.0004], labels=["0", "2", "4"])
ax.set_xticks([0.0, 300.0, 600.0])

# plot mean-field dynamics for intact MG condition
ax = fig.add_subplot(grid[1, 1:])
ax.plot(mf_intact.index, snn_intact["v"], color="black", label="SNN")
ax.plot(mf_intact["v"], color="darkorange", label="MF")
ax.legend()
ax.set_ylabel(r"$v$ (mV)")
ax.set_title(r"(C) MSN network dynamics for intact microglia")

# plot mean-field dynamics for ablated MG condition
ax = fig.add_subplot(grid[2, 1:])
ax.plot(mf_ablated.index, snn_ablated["v"], color="black", label="SNN")
ax.plot(mf_ablated["v"], color="darkorange", label="MF")
ax.legend()
ax.set_ylabel(r"$v$ (mV)")
ax.set_xlabel("time (ms)")
ax.set_title(r"(D) MSN network dynamics for ablated microglia")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/msn_microglia.pdf')
plt.show()
