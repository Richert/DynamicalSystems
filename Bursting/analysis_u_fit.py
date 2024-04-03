import numpy as np
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
from numba import njit
from scipy.optimize import minimize
from scipy.stats import cauchy
from scipy.ndimage import gaussian_filter1d
plt.rcParams['backend'] = 'TkAgg'


# function definitions
######################

def smooth(signal: np.ndarray, window: int = 10):
    # N = len(signal)
    # for start in range(N):
    #     signal_window = signal[start:start+window] if N - start > window else signal[start:]
    #     signal[start] = np.mean(signal_window)
    return gaussian_filter1d(signal, sigma=window)


@njit
def get_kld(ps: np.ndarray, qs: np.ndarray) -> float:
    kld = np.zeros_like(ps)
    n = len(kld)
    for i, (p, q) in enumerate(zip(ps, qs)):
        if p > 0 and q > 0:
            kld[i] = p * np.log(p/q)
        elif p > 0:
            kld[i] = n
    return np.sum(kld)


def multi_cauchy(centers: np.asarray, widths: np.asarray, bins: np.asarray):
    n = len(bins) - 1
    x = np.asarray([(bins[i + 1] + bins[i]) / 2.0 for i in range(n)])
    y_pred = np.zeros((n,))
    for center, width in zip(centers, widths):
        y_pred += cauchy.pdf(x, loc=center, scale=width)
    y_pred /= np.sum(y_pred)
    return x, y_pred


def get_dist(params: np.ndarray, s: np.ndarray, n_bins: int, eval_range: tuple = None, plot: bool = False):

    # determine signal range
    smin, smax = (np.min(s), np.max(s))
    bins = np.linspace(smin, smax, n_bins)

    # estimate empirical distribution
    y, _ = np.histogram(s, bins, density=True)
    y = np.asarray(y, dtype=np.float64)
    y /= np.sum(y)

    # get distribution for current parameters
    n = int(params.shape[0]*0.5)
    centers, widths = params[:n], params[n:]
    x, y_pred = multi_cauchy(centers, widths, bins)

    # compute distance between empirical and Lorentzian distribution
    xmin, xmax = (smin, smax) if eval_range is None else eval_range
    idx = (xmax >= x) * (x >= xmin)
    kld = get_kld(y[idx] / np.sum(y[idx]), y_pred[idx] / np.sum(y_pred[idx]))

    # plot the current fit
    if plot:
        print(f"Centers: {np.round(centers, decimals=2)}")
        print(f"Widths: {np.round(widths, decimals=3)}")
        ymax = 1.2 * np.max(y)
        fig, ax = plt.subplots()
        ax.plot(x, y, label="data")
        ax.plot(x, y_pred, label="fit")
        ax.legend()
        # ax.axvline(x=center - width, ymin=0.0, ymax=1.0, color="red")
        # ax.axvline(x=center + width, ymin=0.0, ymax=1.0, color="red")
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([0.0, ymax])
        ax.set_xlabel("values")
        ax.set_ylabel("p")
        ax.set_title(fr"KLD = {kld}")
        plt.show()

    return kld


def FWHM(s: np.ndarray, plot: bool, m: int = 1, n_bins: int = 500, eval_range: float = None, min_width: float = 1e-6,
         kwargs: dict = None) -> tuple:

    if kwargs is None:
        kwargs = {}

    # get data statistics
    smin, smax = np.min(s), np.max(s)
    max_width = np.abs(smax-smin)
    center = np.mean(s)
    width = max_width*0.1
    coefs = np.zeros((2*m,))
    coefs[:m] += center
    coefs[m:] += width

    if max_width < min_width:
        return coefs, 0.0

    # fit lorentzian
    center_bounds = [(smin, smax)] * m
    width_bounds = [(min_width, max_width)] * m
    res = minimize(get_dist, coefs, args=(s, n_bins, (smin-eval_range, smax+eval_range)),
                   bounds=center_bounds + width_bounds, **kwargs)
    coefs = res.x

    # compute goodness of fit
    kld = get_dist(coefs, s, n_bins, eval_range=(smin-eval_range, smax+eval_range), plot=plot)

    return coefs, kld


def get_fwhm(signal: np.ndarray, pool: Parallel, m: int = 1, n_bins: int = 500, plot_steps: int = 1000,
             eval_range: float = None, min_width: float = 1e-6, **kwargs) -> tuple:
    results = pool(delayed(FWHM)(signal[i, :], (i+1) % plot_steps == 0, m, n_bins, eval_range, min_width, kwargs)
                   for i in range(signal.shape[0]))
    return np.asarray([res[0] for res in results]), np.asarray([res[1] for res in results])


# preparations
##############

# define conditions
conditions = ["no_sfa_1", "no_sfa_2", "weak_sfa_1", "weak_sfa_2", "strong_sfa_1", "strong_sfa_2"]
cond_map = {
        "no_sfa_1": {"kappa": 0.0, "eta": 0.0, "eta_inc": 30.0, "eta_init": -30.0, "b": -5.0, "delta": 5.0},
        "weak_sfa_1": {"kappa": 100.0, "eta": 0.0, "eta_inc": 35.0, "eta_init": 0.0, "b": -5.0, "delta": 5.0},
        "strong_sfa_1": {"kappa": 300.0, "eta": 0.0, "eta_inc": 50.0, "eta_init": 0.0, "b": -5.0, "delta": 5.0},
        "no_sfa_2": {"kappa": 0.0, "eta": -150.0, "eta_inc": 190.0, "eta_init": -50.0, "b": -20.0, "delta": 5.0},
        "weak_sfa_2": {"kappa": 100.0, "eta": -20.0, "eta_inc": 70.0, "eta_init": -100.0, "b": -20.0, "delta": 5.0},
        "strong_sfa_2": {"kappa": 300.0, "eta": 40.0, "eta_inc": 100.0, "eta_init": 0.0, "b": -20.0, "delta": 5.0},
    }

# load simulation data
signals = {}
for cond in conditions:
    signals[cond] = pickle.load(open(f"results/snn_udist_{cond}.pkl", "rb"))["results"]["u"]

# fit recovery variable distributions
#####################################

# parameters to fit
m = 2
mus = np.zeros((m,))
Deltas = np.zeros_like(mus)

# fitting algorithm parameters
method = "Nelder-Mead"
min_width = 1e-10
options = {"maxiter": 500}
tol = 1e-5
n_bins = 200
plot_steps = 10000
eval_range = 100.0

# fitting
results = {}
pool = Parallel(n_jobs=10)
for cond in conditions:
    coefs, klds = get_fwhm(signals[cond], pool=pool, m=m, plot_steps=plot_steps, n_bins=n_bins, eval_range=eval_range,
                           tol=tol, options=options, min_width=min_width, method=method)
    results[cond] = {"centers": np.asarray([coef[:m] for coef in coefs]),
                     "widths": np.asarray([coef[m:] for coef in coefs]),
                     "klds": klds}

# plotting
##########

