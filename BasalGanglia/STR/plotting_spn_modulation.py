import pickle
from pycobi import ODESystem
import matplotlib.pyplot as plt

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# load data
ode = ODESystem.from_file("results/spn_bifurcations.pkl", auto_dir="~/PycharmProjects/auto-07p")
neuron = pickle.load(open("results/spn_singleneuron.pkl", "rb"))
snn = pickle.load(open("results/spn_snn.pkl", "rb"))
mf = pickle.load(open("results/spn_mf.pkl", "rb"))["results"]

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
gs = [2.0, 4.0, 8.0, 16.0]
lines = []
colors = ["black", "red", "blue", "green"]
for i in range(len(gs)):
    l = ode.plot_continuation("p/spn_op/I_ext", "p/spn_op/Delta", cont=f"D/I:{i+1}", ax=ax, line_style_unstable="solid",
                              line_color_stable=colors[i], line_color_unstable=colors[i])
    lines.append(l)
ax.set_xlabel(r"$I (pA)$")
ax.set_ylabel(r"$\Delta$ (mV)")
ax.legend(lines, [fr"$g_i = {g}$ nS" for g in gs])
ax.set_title("(B) 2D bifurcation diagram")

# plot 1D bifurcation diagram
ax = fig.add_subplot(grid[1, 1:])
ode.plot_continuation("p/spn_op/Delta", "p/spn_op/v", cont=f"D:1", ax=ax, line_style_unstable="dotted",
                      line_color_stable="black", line_color_unstable="grey")
ode.plot_continuation("p/spn_op/Delta", "p/spn_op/v", cont=f"D:lc:1", ax=ax, line_style_unstable="dotted",
                      line_color_stable="green", line_color_unstable="green", bifurcation_legend=False)
ax.set_xlim([0.05, 0.45])
ax.set_xlabel(r"$\Delta$ (mV)")
ax.set_ylabel(r"$v$ (mV)")
ax.set_title("(C) 1D bifurcation diagram")

# plot mean-field dynamics
ax = fig.add_subplot(grid[2, 1:])
ax.plot(mf.index, snn["v"], color="black", label="SNN")
ax.plot(mf["v"], color="darkorange", label="MF")
ax.legend()
ax.set_ylabel(r"$v$ (mV)")
ax.set_xlabel("time (ms)")
ax.set_title("(D) MSN network dynamics")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/msn.svg')
plt.show()
