from rectipy import readout
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks, hilbert
from scipy.optimize import minimize


def gaussian(x, mu: float = 0.0, sigma: float = 1.0):
    return np.exp(-0.5*((x - mu)/sigma)**2)


def rmse(x: np.ndarray, y: np.ndarray):
    diff = x - y
    return np.sqrt((diff @ diff.T)/diff.shape[0])


def fit_gaussian(sigma: float, x: np.ndarray):
    y_coords = np.arange(x.shape[0])
    center = int(x.shape[0]/2)
    y_coords -= center
    y_coords[center:] += 1
    y = gaussian(y_coords, sigma=sigma)
    y /= np.max(y)
    return rmse(x, y)


def get_kernel(X: np.ndarray, alpha: float = 1e-12):
    """
    """
    X_t = X.T
    return X @ np.linalg.inv(X_t @ X + alpha*np.eye(X.shape[1])) @ X_t


def get_kernel_peaks(K: np.ndarray, **kwargs):
    x = K[::-1, :][np.eye(K.shape[0]) > 0]
    kmax = np.max(x)
    if kmax <= 0:
        return 0
    x /= np.max(x)
    center = int(len(x) / 2) - 1
    p, _ = find_peaks(x[:center], **kwargs)
    return len(p)/center


def get_kernel_var(K: np.ndarray):
    x = K[::-1, :][np.eye(K.shape[0]) > 0]
    kmax = np.max(x)
    if kmax <= 0:
        return 0
    x /= np.max(x)
    center = int(len(x)/2) - 1
    return np.var(x[:center])


def get_kernel_width(K: np.ndarray, **kwargs):
    x = K[::-1, :][np.eye(K.shape[0]) > 0]
    kmax = np.max(x)
    if kmax <= 0:
        return 0
    # x_hil = hilbert(x)
    # x_env = np.abs(x_hil)
    x /= np.max(x)
    res = minimize(fit_gaussian, x0=np.asarray([1.0]), args=(x,), **kwargs)
    return res.x[0]*res.fun


def get_kernel_diff(K: np.ndarray, **kwargs):
    return np.mean(K - np.eye(K.shape[0]))


# load data
fname = "snn_data"
data = pickle.load(open(f"results/{fname}.pkl", "rb"))

# get system dynamics kernel matrix
###################################

cutoff = 1000
kernels, peaks, vars, widths, diffs = [], [], [], [], []
for d in data["s"]:

    X = d.iloc[cutoff:, :].values
    K = get_kernel(X)
    peaks.append(get_kernel_peaks(K, prominence=0.4, height=0.4))
    vars.append(get_kernel_var(K))
    widths.append(get_kernel_width(K))
    diffs.append(get_kernel_diff(K))
    kernels.append(K)

# save data to file
data["K"] = kernels
data["K_peaks"] = peaks
data["K_var"] = vars
data["K_width"] = widths
data["K_diff"] = diffs
pickle.dump(data, open(f"results/{fname}.pkl", "wb"))

# plotting
##########

for idx in [0, 2, 4]:

    _, ax = plt.subplots()
    ax.imshow(kernels[idx], aspect=1.0)
    plt.title('K')

    plt.show()
