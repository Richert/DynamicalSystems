import pandas as pd
from rectipy import readout
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

# load data
fname = f"ct_delta_1"
path = "results"
data = pickle.load(open(f"{path}/{fname}.pkl", "rb"))
print(f"Condition: {data['sweep']}")


# create target data for different taus
#######################################

# create target data
combination_length = np.asarray([1, 2, 3])
I_ext = data["I_ext"].values
targets = []
for cl in combination_length:
    t = np.ones((I_ext.shape[0],))
    for idx in np.random.randint(low=0, high=I_ext.shape[1], size=cl):
        t *= I_ext[:, idx]
    t[t > 0.5] = 1.0
    t[t <= 0.5] = 0.0
    targets.append(t)

# perform readout for each set of target data
#############################################

# create 2D dataframes
train_scores = pd.DataFrame(columns=combination_length, data=np.zeros((len(data["s"]), len(combination_length))))
test_scores = pd.DataFrame(columns=combination_length, data=np.zeros((len(data["s"]), len(combination_length))))
weights = []
intercepts = []
predictions_plotting = []
targets_plotting = []

# training procedure
cutoff = 1000
plot_length = 2000
offset = 25
for i, signal in enumerate(data["s"]):

    s = signal.iloc[cutoff+offset:, :]

    weights_tmp = []
    intercepts_tmp = []
    predictions_plotting.append([])
    targets_plotting.append([])

    for j, (tau, target) in enumerate(zip(combination_length, targets)):

        # readout training
        res = readout(s, target[cutoff:-offset], alpha=10.0, solver='lsqr', positive=False, tol=1e-4,
                      train_split=15000)
        train_scores.iloc[i, j] = res['train_score']
        test_scores.iloc[i, j] = res['test_score']
        weights_tmp.append(res["readout_weights"])
        intercepts_tmp.append(res["readout_bias"])
        predictions_plotting[-1].append(res["prediction"])
        targets_plotting[-1].append(res["target"])

    weights.append(weights_tmp)
    intercepts.append(intercepts_tmp)

# save data to file
data["combinations"] = combination_length
data["train_scores"] = train_scores
data["test_scores"] = test_scores
data["weights"] = weights
data["intercepts"] = intercepts
pickle.dump(data, open(f"{path}/{fname}.pkl", "wb"))

# plotting
import matplotlib.pyplot as plt

params, values = data["sweep"]
examples = [0, 1, 2]
for trial in range(0, len(data["s"])):

    vals = values[trial]
    title = ", ".join([f"{p} = {v}" for p, v in zip(params, vals)])

    fig, axes = plt.subplots(nrows=2+len(examples), figsize=(10, 8))

    ax = axes[0]
    ax.plot(combination_length, test_scores.iloc[trial, :])
    ax.set_xlabel("phi")
    ax.set_ylabel("test score")
    ax.set_title(title)

    ax = axes[1]
    ax.plot(combination_length, train_scores.iloc[trial, :])
    ax.set_xlabel("phi")
    ax.set_ylabel("train score")

    for ax, ex in zip(axes[2:], examples):
        ax.plot(predictions_plotting[trial][ex], color="blue")
        ax.plot(targets_plotting[trial][ex], color="orange")
        ax.set_xlabel("time")
        ax.set_ylabel("s")
        ax.set_title(f"sigma = {combination_length[ex]}")

    plt.tight_layout()
    plt.show()
