import pandas as pd
from rectipy import readout
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d
import sys
from typing import Tuple, List, Iterable

# load data
fname = f"rs_ir"
data = pickle.load(open(f"results/{fname}_results.pkl", "rb"))
config = pickle.load(open(f"config/{fname}_config.pkl", "rb"))
print(f"Condition: {data['sweep']}")


def get_target(steps: int, times: np.ndarray, channels: List[Iterable], dur: int, func: str, target_channels: Iterable[int]
               ) -> np.ndarray:
    target = np.zeros((steps,))
    for c, t in zip(channels, times):
        criterion = np.sum([tc in c for tc in target_channels])
        if func == "AND":
            response = 1 if criterion == 2 else 0
        elif func == "OR":
            response = 1 if criterion > 0 else 0
        elif func == "XOR":
            response = 1 if criterion == 1 else 0
        else:
            raise ValueError("Wrong target function.")
        target[int(t):int(t)+dur] = response
    return target


# create target data for different taus
#######################################

# get input data
stim_times = np.asarray([t/config["sr"] for t in config["stim_times"]])
stim_channels = config["stim_channels"]
stim_dur = config["stim_dur"]
m = config["W_in"].shape[1]
steps = data["I_ext"].shape[0]

# choose lags at which network response should be read out
lags = np.asarray([0.0, 50.0, 100.0, 150.0, 200.0, 250.0])/(config["dt"]*config["sr"])

# choose functional relationship between input channels that should be picked up
funcs = ["XOR", "XOR", "XOR", "XOR", "XOR", "XOR"]
target_channels = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (0, 2)]

# create target signals
targets = {}
for func, channels in zip(funcs, target_channels):
    targets[(func, channels)] = {}
    for lag in lags:
        readout_times = stim_times + lag
        targets[(func, channels)][lag] = get_target(steps, readout_times, stim_channels, stim_dur, func, channels)

# perform readout for each set of target data
#############################################

# create 2D dataframes
train_scores = pd.DataFrame(columns=funcs, data=np.zeros((len(lags), len(funcs))), index=lags)
test_scores = pd.DataFrame(columns=funcs, data=np.zeros((len(lags), len(funcs))), index=lags)
weights = []
intercepts = []
predictions_plotting = []
targets_plotting = []

# training procedure
cutoff = int(config["cutoff"]/(config["dt"]*config["sr"]))
signal = data["s"].iloc[cutoff:, :].values
train_split = int(0.8*(config["T"] - config["cutoff"])/(config["dt"]*config["sr"]))
for j, (func, channels) in enumerate(zip(funcs, target_channels)):

    weights_tmp = []
    intercepts_tmp = []
    predictions_plotting.append([])
    targets_plotting.append([])

    for i, lag in enumerate(lags):

        target = targets[(func, channels)][lag]

        # readout training
        res = readout(signal, target[cutoff:], alpha=10.0, solver='lsqr', positive=False, tol=1e-4, train_split=train_split)
        train_scores.iloc[i, j] = res['train_score']
        test_scores.iloc[i, j] = res['test_score']
        weights_tmp.append(res["readout_weights"])
        intercepts_tmp.append(res["readout_bias"])
        predictions_plotting[-1].append(res["prediction"])
        targets_plotting[-1].append(res["target"])

    weights.append(weights_tmp)
    intercepts.append(intercepts_tmp)

# save data to file
data["train_scores"] = train_scores
data["test_scores"] = test_scores
data["weights"] = weights
data["intercepts"] = intercepts
pickle.dump(data, open(f"results/{fname}_readouts.pkl", "wb"))

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 8))
grid = GridSpec(nrows=4, ncols=2)
for i, (data, title) in enumerate(zip([train_scores, test_scores], ["train scores", "test_scores"])):
    ax = fig.add_subplot(grid[0, i])
    ax.imshow(data.values, aspect="auto", interpolation="none")
    ax.set_xlabel("funcs")
    ax.set_ylabel("lags")
    ax.set_xticks(np.arange(0, len(funcs), 1), data.columns.values)
    ax.set_yticks(np.arange(0, len(lags), 2), data.index.values[::2])
    ax.set_title(title)

examples = [(0, 0), (1, 0), (2, 0)]
for i, (func, lag) in enumerate(examples):
    ax = fig.add_subplot(grid[i+1, :])
    ax.plot(predictions_plotting[func][lag], color="black", label="prediction")
    ax.plot(targets_plotting[func][lag], color="red", label="target")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("s")
    ax.set_title(f"Func = {funcs[func]}, lag = {lags[lag]}")
    plt.legend()

plt.tight_layout()
plt.show()
