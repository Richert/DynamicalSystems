from rectipy import readout
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d

# load data
fname = "snn_data5"
data = pickle.load(open(f"results/{fname}.pkl", "rb"))
I_ext = data["I_ext"].loc[:, 0]

# create target data for different taus
#######################################

# create time vector
dt = data["dt"]
time = (I_ext.index - np.min(I_ext.index)) * dt
#I_ext = np.random.randn(I_ext.shape[0])

# create target data
taus = [1100.0, 1200.0, 1400.0, 1800.0, 2600.0]
steps = int(I_ext.shape[0]/dt)
sigma, amp = 200.0, np.max(I_ext)
targets = []
for tau in taus:
    I_tmp = np.zeros((steps, 1))
    I_tmp[int(tau/dt)] = amp
    I_tmp = gaussian_filter1d(I_tmp, sigma=sigma, axis=0)
    targets.append(I_tmp[::100])

# perform readout for each set of target data
#############################################

# training procedure
cutoff = 1000
plot_length = 2000
s = data["s"].iloc[cutoff:, :]
scores = []
for tau, target in zip(taus, targets):

    # ridge regression
    res = readout(s, target[cutoff:])

    # plotting
    plt.plot(res["target"][:plot_length], color="black", linestyle="dashed")
    plt.plot(res["prediction"][:plot_length], color="orange")
    plt.legend(["target", "prediction"])
    plt.title(f"tau = {tau}, score = {res['train_score']}")
    plt.show()

    scores.append(res["train_score"])

# save data to file
data["taus"] = taus
data["scores"] = scores
pickle.dump(data, open(f"results/{fname}.pkl", "wb"))
