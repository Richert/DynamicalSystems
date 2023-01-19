import pandas as pd
from rectipy import readout
import numpy as np
import pickle

# load data
fname = f"ir_rs_data"
path = "results"
data = pickle.load(open(f"{path}/{fname}.pkl", "rb"))
I_ext = data["I_ext"]

print(f"Condition: {data['sweep']}")


def sigmoid(x, kappa, t_on, omega):
    return 1.0/(1.0 + np.exp(-kappa*(x-np.cos(t_on*np.pi/omega))))


# create target data for different taus
#######################################

# create target data
phis = np.linspace(0.1, 0.9, num=10)*2.0*np.pi
freq = 0.001
T = data["T"]
steps = int(T/data["dt"])
targets = []
for phi in phis:
    I_tmp = sigmoid(np.cos(np.linspace(0, T, steps)*2.0*np.pi*freq - phi), kappa=5e2, t_on=1.0, omega=1/freq)
    targets.append(I_tmp[::data['sr']])

# perform readout for each set of target data
#############################################

# create 2D dataframes
train_scores = pd.DataFrame(columns=phis, data=np.zeros((len(data["s"]), len(phis))))
test_scores = pd.DataFrame(columns=phis, data=np.zeros((len(data["s"]), len(phis))))
weights = []
intercepts = []
predictions_plotting = []
targets_plotting = []

# training procedure
cutoff = 1000
plot_length = 2000
for i, signal in enumerate(data["s"]):

    s = signal.iloc[cutoff:, :]

    weights_tmp = []
    intercepts_tmp = []
    predictions_plotting.append([])
    targets_plotting.append([])
    for j, (tau, target) in enumerate(zip(phis, targets)):

        # readout training
        res = readout(s, target[cutoff:], alpha=10.0, solver='lsqr', positive=False, tol=0.1, train_split=8000)
        train_scores.iloc[i, j] = res['train_score']
        test_scores.iloc[i, j] = res['test_score']
        weights_tmp.append(res["readout_weights"])
        intercepts_tmp.append(res["readout_bias"])
        predictions_plotting[-1].append(res["prediction"])
        targets_plotting[-1].append(res["target"])

    weights.append(weights_tmp)
    intercepts.append(intercepts_tmp)

# save data to file
data["lags"] = phis
data["train_scores"] = train_scores
data["test_scores"] = test_scores
data["weights"] = weights
data["intercepts"] = intercepts
pickle.dump(data, open(f"{path}/{fname}.pkl", "wb"))

# plotting
import matplotlib.pyplot as plt

params, values = data["sweep"]
for trial in range(0, len(data["s"])):

    vals = values[trial]
    title = ", ".join([f"{p} = {v}" for p, v in zip(params, vals)])

    fig, axes = plt.subplots(nrows=5, figsize=(10, 8))

    ax = axes[0]
    ax.plot(phis, test_scores.iloc[trial, :])
    ax.set_xlabel("phi")
    ax.set_ylabel("test score")
    ax.set_title(title)

    examples = [0, 2, 4, 6]
    for ax, ex in zip(axes[1:], examples):
        ax.plot(predictions_plotting[trial][ex], color="blue")
        ax.plot(targets_plotting[trial][ex], color="orange")
        ax.set_xlabel("time")
        ax.set_ylabel("s")

    plt.tight_layout()
    plt.show()
