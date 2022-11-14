from rectipy import readout
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks


def get_kernel(X: np.ndarray, alpha: float = 1e-12):
    """
    """
    X_t = X.T
    return X @ np.linalg.inv(X_t @ X + alpha*np.eye(X.shape[1])) @ X_t


def get_kernel_width(x: np.ndarray, **kwargs):
    peaks, props = find_peaks(x, **kwargs)
    plt.plot(x)
    plt.vlines(peaks, ymin=np.min(x), ymax=np.max(x), color='orange', linestyle='dashed')
    plt.show()
    h = props["peak_heights"]
    idx = np.argsort(h)
    return np.abs(peaks[idx[-1]] - peaks[idx[-2]])/(h[idx[-1]] - h[idx[-2]])


# load data
fname = "snn_data"
data = pickle.load(open(f"results/{fname}.pkl", "rb"))

# get system dynamics kernel matrix
###################################

cutoff = 1000
kernels, kernel_widths = [], []
for d in data["s"]:

    X = d.iloc[cutoff:, :].values
    K = get_kernel(X)
    off_diag = K[::-1, :][np.eye(K.shape[0]) > 0]
    kmax = np.max(off_diag)
    if kmax > 0:
        off_diag /= kmax
        kernel_width = get_kernel_width(off_diag[:1515], height=0.2, prominence=0.2)
    else:
        kernel_width = 0
    kernels.append(K)
    kernel_widths.append(kernel_width)

# save data to file
data["K"] = kernels
data["K_widths"] = kernel_widths
pickle.dump(data, open(f"results/{fname}.pkl", "wb"))

# plotting
##########

for idx in [0, 2, 4]:

    _, ax = plt.subplots()
    ax.imshow(kernels[idx], aspect=1.0)
    plt.title('K')

    plt.show()
