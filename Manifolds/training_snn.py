from rectipy import readout
import numpy as np
import matplotlib.pyplot as plt
import pickle

# load data
fname = "snn_data3"
data = pickle.load(open(f"results/{fname}.pkl", "rb"))
I_ext = data["I_ext"].loc[:, 0]

# create target data for different taus
#######################################

# create time vector
dt = data["dt"]
time = (I_ext.index - np.min(I_ext.index)) * dt

# create target data
taus = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
epsilon = 1e-5
targets = []
for tau in taus:
    kernel_length = - int(tau * np.log(epsilon))
    kernel = np.exp(-time[:kernel_length]/tau)
    targets.append(np.convolve(I_ext, kernel, mode="same"))

# perform readout for each set of target data
#############################################

# training procedure
cutoff = 1000
plot_length = 1000
s = data["s"].iloc[cutoff:-cutoff, :]
scores = []
for tau, target in zip(taus, targets):

    # ridge regression
    res = readout(s, target[cutoff:-cutoff])

    # plotting
    plt.plot(res["target"][-plot_length:], color="black", linestyle="dashed")
    plt.plot(res["prediction"][-plot_length:], color="orange")
    plt.legend(["target", "prediction"])
    plt.title(f"tau = {tau}, score = {res['train_score']}")
    plt.show()

    scores.append(res["train_score"])

# save data to file
data["taus"] = taus
data["scores"] = scores
pickle.dump(data, open(f"results/{fname}.pkl", "wb"))
