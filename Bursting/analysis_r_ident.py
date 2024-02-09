import numpy as np
from sysidentpy.model_structure_selection import FROLS, AOLS
from sysidentpy.basis_function._basis_function import Polynomial
from pandas import DataFrame
from sysidentpy.utils.display_results import results as sys_results
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'


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

# regress r via other mean-field variables
##########################################
features = ["v", "u", "x"]
results = {cond: {} for cond in conditions}
for cond in conditions:

    data = signals[cond]["snn_etas"]

    # create regressors and target
    X = np.stack([data[f] for f in features], axis=-1)
    for i in range(X.shape[1]):
        # X[:, i] -= np.mean(X[:, i])
        X[:, i] /= np.max(X[:, i])
    y = np.reshape(data["r"], (data["r"].shape[0], 1))

    # fit sindy model to data
    basis_functions = Polynomial(degree=3)
    # model = FROLS(ylag=2, xlag=[[2] for _ in range(X.shape[1])], basis_function=basis_functions, n_terms=4)
    model = AOLS(ylag=1, xlag=[[3] for _ in range(X.shape[1])], basis_function=basis_functions,
                 k=4, threshold=1e-7, L=1)
    model.fit(X=X, y=y)

    # test model
    y_pred = model.predict(X=X, y=y)

    # print results
    res = sys_results(model.final_model, model.theta, model.err, model.n_terms, err_precision=8, dtype="sci")  # type: list
    for i in range(len(res)):
        regressor = res[i][0]
        for j in range(len(features)):
            regressor = regressor.replace(f"x{j + 1}", features[j])
        res[i][0] = regressor
    r = DataFrame(
        res,
        columns=["Regressors", "Parameters", "ERR"],
    )
    print(r)

    # store results
    results[cond]["v"] = data["v"]
    results[cond]["u"] = data["u"]
    results[cond]["y"] = y[:, 0]
    results[cond]["y_pred"] = y_pred[:, 0]
    results[cond]["time"] = signals[cond]["mf_etas_global"]["s"].index

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
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# create figure layout
fig = plt.figure()
grid = fig.add_gridspec(nrows=len(plot_conditions)*2, ncols=2)
plt.suptitle("Lorentzian ansatz II: Population statistics")
for i, cond in enumerate(plot_conditions):

    for j in range(2):

        c = f"{cond}_{j+1}"

        # extract time
        time = results[c]["time"]

        # plot predictors
        ax = fig.add_subplot(grid[i*2, j])
        ax.plot(time, results[c]["v"], color="royalblue")
        ax.set_xlabel("")
        ax.set_ylabel("v (mV)", color="royalblue")
        ax2 = ax.twinx()
        ax2.plot(time, results[c]["u"], color="darkred")
        ax2.set_ylabel("u (pA)", color="darkred")

        # plot target
        ax = fig.add_subplot(grid[i*2+1, j])
        ax.plot(time, results[c]["y"]*1e3, label="target", color="black")
        ax.plot(time, results[c]["y_pred"]*1e3, label="fit", color="darkorange")
        ax.set_ylabel("r (Hz)")
        ax.set_xlabel("time (ms)")
        # if i == 1 and j == 0:
        #     ax.legend()

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/r_fit.pdf')
plt.show()
