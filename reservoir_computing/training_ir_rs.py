import pandas as pd
from rectipy import readout
import numpy as np
import pickle
from typing import List, Iterable

# load data
fname = f"rs_ir2"
data = pickle.load(open(f"results/{fname}_results.pkl", "rb"))
config = pickle.load(open(f"config/{fname}_config.pkl", "rb"))
print(f"Condition: {data['sweep']}")


def get_target(steps: int, times: np.ndarray, channels: Iterable[int], dur: int, n_channels: int = 3
               ) -> np.ndarray:
    target = np.zeros((steps, n_channels))
    for t, c in zip(times, channels):
        target[int(t):int(t)+dur, c] = 1.0
    return target


def wta_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    correct_predictions = []
    idxs = np.argwhere(np.sum(targets, axis=1) > 0).squeeze()
    for idx in idxs:
        correct_predictions.append(np.argmax(targets[idx, :]) == np.argmax(predictions[idx, :]))
    return np.mean(np.asarray(correct_predictions)).squeeze()


# create target data for different taus
#######################################

# get input data
stim_times = np.asarray([t/config["sr"] for t in config["stim_times"]])
stim_channels = config["stim_channels"]
m = config["W_in"].shape[1]
steps = data["I_ext"].shape[0]

# choose lags at which network response should be read out
lags = np.asarray([0.0, 50.0, 100.0, 150.0, 200.0, 250.0])/(config["dt"]*config["sr"])

# create target signals
targets = {}
for lag in lags:
    readout_times = stim_times + lag
    targets[lag] = get_target(steps, readout_times, stim_channels, 50, m)

# perform readout for each set of target data
#############################################

# create 2D dataframes
scores = pd.DataFrame(columns=["train", "test", "wta"], data=np.zeros((len(lags), 3)), index=lags)
weights = []
intercepts = []
predictions_plotting = []
targets_plotting = []

# training procedure
cutoff = int(config["cutoff"]/(config["dt"]*config["sr"]))
signal = data["s"].iloc[cutoff:, :].values
train_split = int(0.8*(config["T"] - config["cutoff"])/(config["dt"]*config["sr"]))

for lag in lags:

    target = targets[lag]

    # readout training
    res = readout(signal, target[cutoff:], alpha=10.0, solver='lsqr', positive=False, tol=1e-4,
                  train_split=train_split)
    scores.loc[lag, "train"] = res['train_score']
    scores.loc[lag, "test"] = res['test_score']
    scores.loc[lag, "wta"] = wta_score(res["target"], res["prediction"])
    weights.append(res["readout_weights"])
    intercepts.append(res["readout_bias"])
    predictions_plotting.append(res["prediction"])
    targets_plotting.append(res["target"])

# save data to file
data["scores"] = scores
data["weights"] = weights
data["intercepts"] = intercepts
pickle.dump(data, open(f"results/{fname}_readouts.pkl", "wb"))

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 8))
grid = GridSpec(nrows=4, ncols=2)
plot_scores = ["test", "wta"]
for i, score in enumerate(plot_scores):
    ax = fig.add_subplot(grid[0, i])
    ax.plot(scores[score].values)
    ax.set_xlabel("lags")
    ax.set_ylabel("score")
    ax.set_xticks(np.arange(0, len(lags), 1), scores.index.values)
    ax.set_title(score)

target_lag = 0
for i in range(m):
    ax = fig.add_subplot(grid[i+1, :])
    ax.plot(predictions_plotting[target_lag][:, i], color="black", label="prediction")
    ax.plot(targets_plotting[target_lag][:, i], color="red", label="target")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(f"Activation of channel #{i+1}")
    if i == 0:
        ax.set_title(f"Lag = {lags[target_lag]}")
    plt.legend()

plt.tight_layout()
plt.show()
