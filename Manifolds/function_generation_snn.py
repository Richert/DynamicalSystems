from rectipy import readout
import numpy as np
import matplotlib.pyplot as plt
import pickle


def get_kernel(X: np.ndarray):
    """
    """
    X_t = X.T
    return X_t @ np.linalg.pinv(X @ X_t) @ X


# load data
fname = "snn_data5"
data = pickle.load(open(f"results/{fname}.pkl", "rb"))

# get system dynamics kernel matrix
###################################

cutoff = 1000
K = get_kernel(data["s"].iloc[cutoff:, :].values.T)
I_ext = data["I_ext"].iloc[cutoff:, :]

# plotting
##########

for comp in [100, 200, 400, 800]:
    signal = K.T[comp, :] @ I_ext
    plt.plot(signal)
    plt.show()

# # save data to file
# data["taus"] = taus
# data["scores"] = scores
# pickle.dump(data, open(f"results/{fname}.pkl", "wb"))
