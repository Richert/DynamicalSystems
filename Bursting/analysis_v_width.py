import numpy as np
from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Lasso
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

# system identification
#######################

results = {cond: {} for cond in conditions}
features = ["r", "v", "u", "x", "u_width"]

for cond in conditions:

    data = signals[cond]["snn_etas"]

    # create input and output data
    X = np.stack([data[f] for f in features], axis=-1)
    for i in range(X.shape[1]):
        # X[:, i] -= np.min(X[:, i])
        X[:, i] /= np.max(X[:, i])
    y = np.reshape(data["v_width"], (data["v_width"].shape[0], 1))

    # initialize system identification model
    model = Lasso(alpha=0.1, fit_intercept=True, max_iter=1000, tol=1e-10)

    # fit model
    model.fit(X=X, y=y)

    # predict width of u
    predictions = model.predict(X=X)
    var_explained = [explained_variance_score(y, c*X[:, i]) for i, c in enumerate(model.coef_)]

    # extract results
    print(f"Condition: {cond}")
    print(f"Model fit: {[f'{f} = {c}' for f, c in zip(features, model.coef_)]}")

    # store results
    results[cond]["v_width_pred"] = predictions
    for f in features:
        results[cond][f] = data[f]
    results[cond]["v_width"] = data["v_width"]
    results[cond]["v_width_mf"] = np.pi*C*signals[cond]["mf_etas_global"]["s"]/(k*tau_s)
    results[cond]["var_explained"] = var_explained

# plotting
##########

# features to plot
plot_features = ["r", "u"]
plot_conditions = ["no_sfa", "weak_sfa", "strong_sfa"]

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# create figure layout
fig = plt.figure()
grid = fig.add_gridspec(nrows=2*(len(plot_features) + 1), ncols=len(plot_conditions))

for j, cond in enumerate(plot_conditions):

    # example 1
    ###########

    c1 = f"{cond}_1"

    # extract time
    time = results[c1]["v_width_mf"].index

    # plot mean-field regressors
    for i, f in enumerate(plot_features):
        ax = fig.add_subplot(grid[i, j])
        ax.plot(time, results[c1][f])
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(f)
        idx = features.index(f)
        v_explained = results[c1]['var_explained'][idx]
        ax.set_title(f"Variance explained: {np.round(v_explained, decimals=2)}")

    # plot target
    ax = fig.add_subplot(grid[len(plot_features), j])
    ax.plot(time, results[c1]["v_width"], label="target", color="black")
    ax.plot(time, results[c1]["v_width_pred"], label="fit", color="darkorange")
    ax.set_ylabel("width(v)")
    ax.set_xlabel("time (ms)")
    ax.set_title("Mean-field prediction of width(v)")

    # example 2
    ###########

    c2 = f"{cond}_2"
    row0 = len(plot_features) + 1

    # plot mean-field regressors
    for i, f in enumerate(plot_features):
        ax = fig.add_subplot(grid[row0 + i, j])
        ax.plot(time, results[c2][f])
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(f)
        idx = features.index(f)
        v_explained = results[c2]['var_explained'][idx]
        ax.set_title(f"Variance explained: {np.round(v_explained, decimals=2)}")

    # plot target
    ax = fig.add_subplot(grid[row0 + len(plot_features), j])
    ax.plot(time, results[c2]["v_width"], label="target", color="black")
    ax.plot(time, results[c2]["v_width_pred"], label="fit", color="darkorange")
    ax.set_ylabel("width(v)")
    ax.set_xlabel("time (ms)")
    if j == 1:
        ax.legend()
    ax.set_title("Mean-field prediction of width(v)")

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/v_width.pdf')
plt.show()
