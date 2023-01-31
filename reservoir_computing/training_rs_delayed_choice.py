import pandas as pd
from rectipy import readout
import numpy as np
import pickle
from typing import Iterable
from sklearn.preprocessing import label_binarize
import sys

# load data
cond = 1 #sys.argv[-1]
fname = f"rs_dc_{cond}"
data = pickle.load(open(f"results/{fname}_results.pkl", "rb"))
config = pickle.load(open(f"config/{fname}_config.pkl", "rb"))
print(f"Condition: {config['sweep']}")


def get_target(steps: int, times: np.ndarray, channels: Iterable[int], dur: int, n_channels: int = 3
               ) -> np.ndarray:
    target = np.zeros((steps, n_channels + 1))
    for t, c in zip(times, channels):
        target[int(t):int(t)+dur, c+1] = 1.0
    target[np.sum(target, axis=1) < 1.0, 0] = 1.0
    return target


def wta_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    correct_predictions = []
    idxs = np.argwhere(np.sum(targets, axis=1) > 0).squeeze()
    for idx in idxs:
        correct_predictions.append(np.argmax(targets[idx, :]) == np.argmax(predictions[idx, :]))
    return np.mean(np.asarray(correct_predictions)).squeeze()


def wta_predictions(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    wta_preds = np.zeros_like(targets)
    idxs = np.argwhere(np.sum(targets, axis=1) > 0).squeeze()
    for row in idxs:
        col = np.argmax(predictions[row, :]).squeeze()
        wta_preds[row, col] = 1.0
    return wta_preds


# create target data for different taus
#######################################

# get input data
stim_times = np.asarray([t/config["sr"] for t in config["stim_times"]])
stim_channels = config["stim_channels"]
m = config["W_in"].shape[1]
steps = data["I_ext"].shape[0]

# choose lags at which network response should be read out
lags = np.asarray([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])/(config["dt"]*config["sr"])

# create target signals
targets = {}
for lag in lags:
    readout_times = stim_times + lag
    targets[lag] = get_target(steps, readout_times, stim_channels, 20, m)

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

    target = targets[lag][cutoff:]

    # readout training
    res = readout(signal, np.argmax(target, axis=1), method="LogisticRegression", penalty="l2", l1_ratio=None,
                  class_weight="balanced", solver="lbfgs", n_jobs=8)
    pred = label_binarize(res["prediction"], classes=[i for i in range(m+1)])
    scores.loc[lag, "train"] = res['train_score']
    scores.loc[lag, "test"] = res['test_score']
    scores.loc[lag, "wta"] = wta_score(target[:, 1:], pred)
    weights.append(res["readout_weights"])
    intercepts.append(res["readout_bias"])
    predictions_plotting.append(pred)
    targets_plotting.append(target[:, 1:])

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
wta_preds = wta_predictions(targets_plotting[target_lag], predictions_plotting[target_lag])
for i in range(m):
    ax = fig.add_subplot(grid[i+1, :])
    ax.plot(predictions_plotting[target_lag][:, i], color="black", label="channel activity")
    ax.plot(targets_plotting[target_lag][:, i], color="orange", label="target")
    ax.plot(wta_preds[:, i], color="blue", label="prediction", linestyle="--")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(f"channel # {i}")
    if i == 0:
        ax.set_title(f"Lag = {lags[target_lag]}")
    plt.legend()

plt.tight_layout()
plt.show()
