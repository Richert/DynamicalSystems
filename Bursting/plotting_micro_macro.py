import pickle
import matplotlib.pyplot as plt

# load simulation data
signals = {}
conditions = ["low_delta", "med_delta", "high_delta"]
for cond in conditions:
    signals[cond] = {}
    for model in ["mf", "snn"]:
        data = pickle.load(open(f"results/{model}_etas_{cond}.pkl", "rb"))["results"]
        signals[cond][model] = data

# plot comparison between snn and mf simulation results over three conditions
#############################################################################

# prepare figure
plot_variables = ["v", "s", "x", "u", "u_width", "v_width"]
fig = plt.figure(figsize=(4*len(conditions), len(plot_variables)), dpi=130)
grid = fig.add_gridspec(nrows=len(plot_variables), ncols=len(conditions))

# plotting
for i, cond in enumerate(conditions):
    for j, var in enumerate(plot_variables):

        mf_data = signals[cond]["mf"][var]
        snn_data = signals[cond]["snn"][var]

        ax = fig.add_subplot(grid[j, i])
        ax.plot(snn_data, color="black", label="snn")
        ax.plot(mf_data, color="darkorange", label="mf")
        ax.set_ylabel(var)
        if j == len(plot_variables) - 1:
            ax.set_xlabel("time")
            ax.legend()

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/micro_macro.pdf')
plt.show()
