import pickle
import matplotlib.pyplot as plt

# load simulation data
signals = {}
conditions = ["no_sfa_1", "no_sfa_2", "weak_sfa_1", "weak_sfa_2", "strong_sfa_1", "strong_sfa_2"]
for cond in conditions:
    signals[cond] = {"mf": pickle.load(open(f"results/mf_bs_corrected_{cond}.pkl", "rb"))["results"],
                     "snn": pickle.load(open(f"results/snn_bs_{cond}.pkl", "rb"))["results"]}

# plot comparison between snn and mf simulation results over three conditions
#############################################################################

# prepare figure
plot_variables = ["s", "u"]
fig = plt.figure(figsize=(12, 8), dpi=130)
grid = fig.add_gridspec(nrows=len(conditions), ncols=len(plot_variables))

# plotting
for i, cond in enumerate(conditions):
    for j, var in enumerate(plot_variables):

        mf_data = signals[cond]["mf"][var]
        snn_data = signals[cond]["snn"][var]

        ax = fig.add_subplot(grid[i, j])
        ax.plot(mf_data.index, snn_data, color="black", label="snn")
        ax.plot(mf_data, color="darkorange", label="mf")
        ax.set_ylabel(var)
        if i == len(plot_variables) - 1:
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
