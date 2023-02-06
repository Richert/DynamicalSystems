import pandas as pd
from rectipy import readout
import numpy as np
import pickle
from typing import Iterable
import sys
from sklearn.preprocessing import label_binarize

# load data
cond = sys.argv[-4]
task = sys.argv[-3]
pop_type = sys.argv[-2]
res_dir = sys.argv[-1]
fname = f"{pop_type}_{task}_{cond}"
data = pickle.load(open(f"{res_dir}/{fname}_results.pkl", "rb"))
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
    wta = 0.0
    classes = np.unique(targets)
    for c in classes:
        samples = targets == c
        wta += np.mean([targets[idx] == predictions[idx] for idx in np.argwhere(samples)])
    return wta/len(classes)


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
classes = [i for i in range(m+1)]
for lag in lags:

    target = np.argmax(targets[lag][cutoff:], axis=1)

    # readout training
    res = readout(signal, target, loss="log_loss", penalty="elasticnet", l1_ratio=0.5, alpha=1e-4, tol=1e-6,
                  n_jobs=12, class_weight="balanced", learning_rate="adaptive", eta0=1e-3, test_size=0.2,
                  normalize="standard", shuffle=False)
    scores.loc[lag, "train"] = res['train_score']
    scores.loc[lag, "test"] = res['test_score']
    scores.loc[lag, "wta"] = wta_score(res["target"], res["prediction"])
    weights.append(res["readout_weights"])
    intercepts.append(res["readout_bias"])
    predictions_plotting.append(label_binarize(res["prediction"], classes=classes))
    targets_plotting.append(label_binarize(res["target"], classes=classes))

# save data to file
data["scores"] = scores
data["weights"] = weights
data["intercepts"] = intercepts
pickle.dump(data, open(f"results/{fname}_readouts.pkl", "wb"))

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 8))
grid = GridSpec(nrows=5, ncols=2)
plot_scores = ["train", "wta"]
for i, score in enumerate(plot_scores):
    ax = fig.add_subplot(grid[0, i])
    ax.plot(scores[score].values)
    ax.set_xlabel("lags")
    ax.set_ylabel("score")
    ax.set_xticks(np.arange(0, len(lags), 1), scores.index.values)
    ax.set_title(score)

target_lag = 2
for i in range(m+1):
    ax = fig.add_subplot(grid[i+1, :])
    ax.plot(predictions_plotting[target_lag][:, i], color="black", label="prediction")
    ax.plot(targets_plotting[target_lag][:, i], color="orange", label="target", linestyle="--")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(f"channel # {i}")
    if i == 0:
        ax.set_title(f"Lag = {lags[target_lag]}")
    plt.legend()

plt.tight_layout()
plt.show()
