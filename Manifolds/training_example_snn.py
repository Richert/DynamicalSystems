import pandas as pd
from rectipy import readout
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d

# load data
fname = "snn_example_data"
data = pickle.load(open(f"results/{fname}.pkl", "rb"))
I_ext = data["I_ext"]

# create target data for different taus
#######################################

# create target data
combinations = [([0, 2], "mult"), ([1, 2], "mult"), ([0, 1, 2], "mult")]
targets = []
for (neurons, mode) in combinations:
    if mode == "sum":
        targets.append(np.sum(I_ext.iloc[:, neurons].v1, axis=1))
    else:
        targets.append(np.prod(I_ext.iloc[:, neurons].v1, axis=1))

# perform readout for each set of target data
#############################################

# create 2D dataframe
var, params = data["sweep"]
scores = pd.DataFrame(columns=np.arange(0, len(combinations)), index=params,
                      data=np.zeros((len(params), len(combinations))))

# training procedure
cutoff = 1000
plot_length = 2000
for i, signal in enumerate(data["s"]):

    s = signal.iloc[cutoff:, :]

    for j, target in enumerate(targets):

        # readout training
        res = readout(s, target[cutoff:], train_split=4500)
        scores.iloc[i, j] = res['test_score']

        # plotting
        plt.plot(res["target"][:plot_length], color="black", linestyle="dashed")
        plt.plot(res["prediction"][:plot_length], color="orange")
        plt.legend(["target", "prediction"])
        plt.title(f"score = {res['test_score']}")
        plt.show()

# save data to file
data["combinations"] = combinations
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
ax.set_xticks(np.arange(len(combinations)))
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
ax.set_ylabel("dim", color="blue")
ax2.set_ylabel("score", color="orange")
ax.set_title("dimensionality vs. training score")

plt.tight_layout()
plt.show()
