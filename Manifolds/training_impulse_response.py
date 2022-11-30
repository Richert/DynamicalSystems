import pandas as pd
from rectipy import readout
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d

# load data
fname = "ir_rs_data4"
data = pickle.load(open(f"results/{fname}.pkl", "rb"))
I_ext = data["I_ext"].loc[:, 0].values


def sigmoid(x, kappa, t_on, omega):
    return 1.0/(1.0 + np.exp(-kappa*(x-np.cos(t_on*np.pi/omega))))


# create target data for different taus
#######################################

# create target data
phis = np.linspace(0.1, 0.9, num=7)*2.0*np.pi
freq = 0.002
T = I_ext.shape[0]
steps = int(T/data["dt"])
targets = []
for phi in phis:
    I_tmp = sigmoid(np.cos(np.linspace(0, T, steps)*2.0*np.pi*freq - phi), kappa=1e3, t_on=1.0, omega=1/freq)
    # plt.plot(I_tmp[::100]*50.0)
    # plt.plot(I_ext, color='black', linestyle='--')
    # plt.show()
    targets.append(I_tmp[::100])

# perform readout for each set of target data
#############################################

# create 2D dataframe
var, params = data["sweep"]
scores = pd.DataFrame(columns=phis, index=params, data=np.zeros((len(params), len(phis))))

# training procedure
cutoff = 1000
plot_length = 1000
for i, signal in enumerate(data["s"]):

    s = signal.iloc[cutoff:, :]

    for j, (tau, target) in enumerate(zip(phis, targets)):

        # readout training
        res = readout(s, target[cutoff:], alpha=10.0, solver='lbfgs', positive=True, tol=0.01, train_split=9000)
        res2 = readout(s, target[cutoff:], alpha=10.0, solver='lbfgs', positive=True, tol=0.01)
        res3 = readout(s[:-cutoff], target[cutoff:-cutoff], alpha=10.0, solver='lbfgs', positive=True, tol=0.01)
        weight_diff = res2["readout_weights"] - res3["readout_weights"]
        scores.iloc[i, j] = res['test_score']

        # plotting
        plt.plot(res["target"][-plot_length:], color="black", linestyle="dashed")
        plt.plot(res["prediction"][-plot_length:], color="orange")
        plt.plot((s @ weight_diff + res["readout_bias"]).values[-plot_length:], color="purple")
        plt.legend(["target", "prediction", "new"])
        plt.title(f"tau = {tau}, score = {res['train_score']}")
        plt.show()

# save data to file
data["taus"] = phis
data["scores"] = scores
pickle.dump(data, open(f"results/{fname}.pkl", "wb"))

# plotting
##########

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 6))

# training scores
ax = axes[0, 0]
im = ax.imshow(scores)
ax.set_ylabel(var)
ax.set_yticks(np.arange(len(params)), labels=params)
ax.set_xlabel("tau")
ax.set_xticks(np.arange(len(phis)), labels=phis)
ax.set_title("Training Scores")
plt.colorbar(im, ax=ax)

# average training scores vs. kernel peaks
ax = axes[0, 1]
k = data["K_diff"]
ax.plot(k, color="blue")
ax2 = ax.twinx()
ax2.plot(np.mean(scores.values, axis=1), color="orange")
ax.set_xlabel(var)
ax.set_xticks(np.arange(len(params)), labels=params)
ax.set_ylabel("diff", color="blue")
ax2.set_ylabel("score", color="orange")
ax.set_title("kernel diff vs. training score")

# average training scores vs. kernel variance
ax = axes[1, 0]
k = data["K_var"]
ax.plot(k, color="blue")
ax2 = ax.twinx()
ax2.plot(np.mean(scores.values, axis=1), color="orange")
ax.set_xlabel(var)
ax.set_xticks(np.arange(len(params)), labels=params)
ax.set_ylabel("var", color="blue")
ax2.set_ylabel("score", color="orange")
ax.set_title("K variance vs. training score")

# average training scores vs. kernel width
ax = axes[1, 1]
k = data["X_dim"]
ax.plot(k, color="blue")
ax2 = ax.twinx()
ax2.plot(np.mean(scores.values, axis=1), color="orange")
ax.set_xlabel(var)
ax.set_xticks(np.arange(len(params)), labels=params)
ax.set_ylabel("dims", color="blue")
ax2.set_ylabel("score", color="orange")
ax.set_title("dimensionality vs. training score")

plt.tight_layout()
plt.show()
