import pandas as pd
from rectipy import readout
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d

# load data
fname = "snn_data"
data = pickle.load(open(f"results/{fname}.pkl", "rb"))
I_ext = data["I_ext"].loc[:, 0]

# create target data for different taus
#######################################

# create target data
taus = [2.0, 4.0, 8.0, 16.0]
steps = int(I_ext.shape[0]/data["dt"])
targets = []
for tau in taus:
    I_tmp = gaussian_filter1d(I_ext, sigma=tau, axis=0)
    targets.append(I_tmp)

# perform readout for each set of target data
#############################################

# create 2D dataframe
var, params = data["sweep"]
scores = pd.DataFrame(columns=taus, index=params, data=np.zeros((len(params), len(taus))))

# training procedure
cutoff = 1000
plot_length = 2000
for i, signal in enumerate(data["s"]):

    s = signal.iloc[cutoff:, :]

    for j, (tau, target) in enumerate(zip(taus, targets)):

        # readout training
        res = readout(s, target[cutoff:], train_split=2500)
        scores.iloc[i, j] = res['test_score']

        # plotting
        # plt.plot(res["target"][:plot_length], color="black", linestyle="dashed")
        # plt.plot(res["prediction"][:plot_length], color="orange")
        # plt.legend(["target", "prediction"])
        # plt.title(f"tau = {tau}, score = {res['test_score']}")
        # plt.show()

# save data to file
data["taus"] = taus
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
ax.set_xticks(np.arange(len(taus)), labels=taus)
ax.set_title("Training Scores")
plt.colorbar(im, ax=ax)

# average training scores vs. kernel peaks
ax = axes[0, 1]
k = data["K_peaks"]
ax.plot(k, color="blue")
ax2 = ax.twinx()
ax2.plot(np.mean(scores.values, axis=1), color="orange")
ax.set_xlabel(var)
ax.set_xticks(np.arange(len(params)), labels=params)
ax.set_ylabel("peaks", color="blue")
ax2.set_ylabel("score", color="orange")
ax.set_title("kernel peaks vs. training score")

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
k = data["K_width"]
ax.plot(k, color="blue")
ax2 = ax.twinx()
ax2.plot(np.mean(scores.values, axis=1), color="orange")
ax.set_xlabel(var)
ax.set_xticks(np.arange(len(params)), labels=params)
ax.set_ylabel("width", color="blue")
ax2.set_ylabel("score", color="orange")
ax.set_title("K width vs. training score")

plt.tight_layout()
plt.show()
