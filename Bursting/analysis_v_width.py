import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'


def smooth(signal: np.ndarray, window: int = 10):
    # N = len(signal)
    # for start in range(N):
    #     signal_window = signal[start:start+window] if N - start > window else signal[start:]
    #     signal[start] = np.mean(signal_window)
    return gaussian_filter1d(signal, sigma=window)


# define conditions
conditions = ["no_sfa_1", "no_sfa_2", "weak_sfa_1", "weak_sfa_2", "strong_sfa_1", "strong_sfa_2"]
C = 100.0
k = 0.7
tau_s = 6.0

# load simulation data
signals = {}
mf_models = ["mf_etas_global"]
snn_models = ["snn_etas"]
models = mf_models + snn_models
for cond in conditions:
    signals[cond] = {}
    for model in models:
        data = pickle.load(open(f"results/{model}_{cond}.pkl", "rb"))["results"]
        signals[cond][model] = data

# compare x against width(v)
############################

results = {cond: {} for cond in conditions}
for cond in conditions:

    data = signals[cond]["snn_etas"]

    # calculate explained variance
    x = data["r"]*C*np.pi/k
    y = data["v_width"]
    var_explained = explained_variance_score(y, x)

    # store results
    results[cond]["x"] = x
    results[cond]["v_width"] = y
    results[cond]["var_explained"] = var_explained
    results[cond]["time"] = signals[cond]["mf_etas_global"]["s"].index
    results[cond]["kld"] = data["v_errors"]

# plotting
##########

# conditions to plot
plot_conditions = ["no_sfa", "weak_sfa", "strong_sfa"]

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# create figure layout
fig = plt.figure()
grid = fig.add_gridspec(nrows=len(plot_conditions), ncols=2)
plt.suptitle("Lorentzian ansatz II: Population statistics")

for i, cond in enumerate(plot_conditions):

    for j in range(2):

        c = f"{cond}_{j+1}"

        # extract time
        time = results[c]["time"]

        # plot target
        ax = fig.add_subplot(grid[i, j])
        ax2 = ax.twinx()
        l1, = ax2.plot(time, results[c]["kld"], color="darkred", alpha=0.5)
        ax2.set_ylim([0.0, 1.0])
        l2, = ax.plot(time, results[c]["v_width"], color="black")
        l3, = ax.plot(time, results[c]["x"], color="royalblue")
        ax2.set_ylabel("KLD", color="darkred")
        ax.set_ylabel("x (mV)")
        ax.set_xlabel("time (ms)")
        if i == 1 and j == 0:
            fig.legend([l2, l3, l1], [r"$x(t)$", r"$\frac{C \pi}{k} r(t)$", "KLD"])

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/v_width.pdf')
plt.show()
