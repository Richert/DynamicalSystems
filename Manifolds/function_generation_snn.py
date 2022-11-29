import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks
from scipy.optimize import minimize


def get_dim(s: np.ndarray):
    s -= np.mean(s)
    s /= np.std(s)
    cov = s.T @ s
    cov[np.eye(cov.shape[0]) > 0] = 0.0
    eigs = np.abs(np.linalg.eigvals(cov))
    return np.sum(eigs)**2/np.sum(eigs**2)


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
    x /= np.max(x)
    res = minimize(fit_gaussian, x0=np.asarray([1.0]), args=(x,), **kwargs)
    return res.x[0]*res.fun


def get_kernel_diff(K: np.ndarray, **kwargs):
    return np.mean(K - np.eye(K.shape[0]))


# load data
fname = "ir_rs_data2"
data = pickle.load(open(f"results/{fname}.pkl", "rb"))

# get system dynamics kernel matrix
###################################

cutoff = 900
kernels, vars, diffs, dims = [], [], [], []
for d in data["s"]:

    d = d.iloc[cutoff:, :].values
    inp = data["I_ext"].iloc[cutoff:, 0].values
    peaks, _ = find_peaks(inp)
    isi = peaks[1] - peaks[0]

    kernels_tmp, vars_tmp, diffs_tmp, dims_tmp = [], [], [], []
    for idx in range(len(peaks)):
        if peaks[idx]+isi < d.shape[0]:
            X = d[peaks[idx]:peaks[idx]+isi]
            K = get_kernel(X)
            vars_tmp.append(get_kernel_var(K))
            dims_tmp.append(get_dim(X))
            diffs_tmp.append(get_kernel_diff(K))
            kernels_tmp.append(K)

    kernels.append(np.mean(kernels_tmp, axis=0))
    vars.append(np.mean(vars_tmp))
    diffs.append(np.mean(diffs_tmp))
    dims.append(np.mean(dims_tmp))

# save data to file
data["K"] = kernels
data["K_var"] = vars
data["X_dim"] = dims
data["K_diff"] = diffs
pickle.dump(data, open(f"results/{fname}.pkl", "wb"))

# plotting
##########

for k in kernels:

    _, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].imshow(k, aspect=1.0, cmap='nipy_spectral')
    ax[0].set_title('K')
    ax[1].plot(k[::-1, :][np.eye(k.shape[0]) > 0])
    ax[1].set_title("off diag of K")
    plt.show()
