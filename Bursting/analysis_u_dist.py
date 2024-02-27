import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import cauchy
from typing import Union
from joblib import Parallel, delayed
from numba import njit
from scipy.optimize import minimize

# function definitions
######################


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


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


def get_dist(width: np.ndarray, s: np.ndarray, n_bins: int, eval_range: float = None, plot: bool = False):

    # determine signal range
    smin, smax = (np.min(s), np.max(s))
    bins = np.linspace(smin, smax, n_bins)

    # estimate empirical distribution
    y, _ = np.histogram(s, bins, density=True)
    y = np.asarray(y, dtype=np.float64)

    # get Lorentzian distribution for current parameters
    x = np.asarray([(bins[i + 1] + bins[i]) / 2.0 for i in range(n_bins - 1)])
    center = x[np.argmax(y).squeeze()]
    y_pred = cauchy.pdf(x, loc=center, scale=width)

    # compute distance between empirical and Lorentzian distribution
    xmin, xmax = (smin, smax) if eval_range is None else (center - eval_range, center + eval_range)
    idx = (xmax >= x) * (x >= xmin)
    kld = get_kld(y[idx] / np.sum(y[idx]), y_pred[idx] / np.sum(y_pred[idx]))

    # plot the current fit
    if plot and kld > 0.5:
        ymax = 1.2 * np.max(y)
        xrange = np.max(bins) - np.min(bins)
        fig, ax = plt.subplots()
        ax.plot(x, y, label="data")
        ax.plot(x, y_pred, label="fit")
        ax.legend()
        ax.axvline(x=center - width, ymin=0.0, ymax=1.0, color="red")
        ax.axvline(x=center + width, ymin=0.0, ymax=1.0, color="red")
        ax.set_xlim([center - xrange / 6, center + xrange / 6])
        ax.set_ylim([0.0, ymax])
        ax.set_xlabel("values")
        ax.set_ylabel("p")
        ax.set_title(fr"$\Delta = {width}$, KLD = {kld}")
        plt.show()

    return kld


def FWHM(s: np.ndarray, plot: bool, n_bins: int = 500, eval_range: float = None, min_width: float = 1e-6,
         kwargs: dict = None) -> tuple:

    if kwargs is None:
        kwargs = {}

    # specify upper boundary
    max_width = np.abs(np.max(s)-np.min(s))

    if max_width < min_width:

        width = min_width
        kld = 0.0

    else:

        # fit lorentzian
        res = minimize(get_dist, np.asarray([np.std(s)]), args=(s, n_bins, eval_range),
                       bounds=[(min_width, max_width)], **kwargs)
        width = res.x[0]

        # compute goodness of fit
        kld = get_dist(width, s, n_bins, eval_range=eval_range, plot=plot)

    return width, kld


def get_fwhm(signal: np.ndarray, pool: Parallel, n_bins: int = 500, plot_steps: int = 1000, eval_range: float = None,
             min_width: float = 1e-6, **kwargs) -> tuple:
    results = pool(delayed(FWHM)(signal[i, :], (i+1) % plot_steps == 0, n_bins, eval_range, min_width, kwargs)
                   for i in range(signal.shape[0]))
    return np.asarray([res[0] for res in results]), np.asarray([res[1] for res in results])


def u_dot(t: float, u: np.ndarray, v: np.ndarray, x: np.ndarray, b: Union[float, np.ndarray], v_r: float, tau_u: float,
          kappa: float, mu: Union[float, np.ndarray]) -> float:
    return (b*(v - v_r + mu) - u) / tau_u + kappa*x


def solve(v: np.ndarray, x: np.ndarray, t_init: float, T: float, dt: float, v_r: float, b: Union[float, np.ndarray],
          tau_u: float, kappa: float, mu: Union[float, np.ndarray], n_bins: int = 100, **kwargs) -> dict:

    t_eval = np.linspace(0.0, T, num=int(T/dt))

    # integrate u
    u0 = np.zeros((v.shape[0],))
    u = solve_ivp(u_dot, (0.0, T), u0, args=(v, x, b, v_r, tau_u, kappa, mu), t_eval=t_eval, **kwargs)

    # remove initial condition
    u = np.squeeze(u.y)[:, t_eval >= t_init]

    # estimate distribution of u
    eval_range = 100.0
    u_dist = []
    u_edges = []
    for i in range(u.shape[1]):
        center = np.mean(u[:, i])
        bins = np.linspace(center - eval_range, center + eval_range, num=n_bins+1)
        u_hist, u_edges = np.histogram(u[:, i], bins=bins)
        u_dist.append(u_hist)
    u_edges = [(u_edges[i] + u_edges[i+1])/2 for i in range(len(u_edges)-1)]

    # fit lorentzian distribution to the u distributions
    pool = Parallel(n_jobs=10)
    u_widths, u_errors = get_fwhm(u.T, pool=pool, plot_steps=100000, n_bins=n_bins, eval_range=eval_range,
                                  tol=1e-3, options={"maxiter": 500}, min_width=1e-10, method="Nelder-Mead")

    # collect results
    res = {"u": u, "time": t_eval[t_eval >= t_init], "u_width": u_widths, "kld": u_errors,
           "u_edges": u_edges, "u_dist": np.asarray(u_dist)}

    return res


# analysis
##########

# define model parameters
N = 2000
C = 100.0
k = 0.7
v_r = -60.0
b = -5.0
tau_u = 35.0
mu = 0.0
Delta = 1.0
lam = 1.0
kappa = 0.0

# define quenched constants
mus = mu + Delta * np.tan(0.5*np.pi*(2*np.arange(1, N+1)-N-1)/(N+1))
kappas = kappa + lam * np.tan(0.5*np.pi*(2*np.arange(1, N+1)-N-1)/(N+1))

# define simulation parameters
T = 1100.0
t0 = 100.0
dt = 0.01
method = "RK23"
atol = 1e-9
rtol = 1e-9
steps = int(T/dt)

# define initial state
r = 0.01
v0 = -57.0
x0 = 0.01
v = v0 + r*np.pi*C/k * np.tan(0.5*np.pi*(2*np.arange(1, N+1)-N-1)/(N+1))
x = np.zeros_like(v) + 0.01

# define histogram parameters
n_bins = 500

# simulations
#############

res = solve(v, x, t0, T,  dt, v_r=v_r, b=b, tau_u=tau_u, kappa=kappas, mu=mus, n_bins=n_bins, method=method,
            atol=atol, rtol=rtol)

# plotting
##########

# plot settings
plt.rcParams['backend'] = "TkAgg"
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0

# create figure layout
fig = plt.figure(layout="constrained")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, hspace=0.01, wspace=0.01)
grid = fig.add_gridspec(nrows=4, ncols=1)
fig.suptitle("Lorentzian ansatz IV: Recovery Variable Distribution")
stop = int(1000.0 / dt)

# plot u dynamics
ax = fig.add_subplot(grid[0, 0])
u, time = res["u"], res["time"]
ax.plot(time[:stop], np.mean(u[:, :stop], axis=0), color="royalblue")
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$u$ pA")

# plot u distribution over time
ax = fig.add_subplot(grid[1, 0])
u_edges, u_dist = res["u_edges"], res["u_dist"]
ax.imshow(u_dist[:stop, :].T, aspect="auto", cmap="viridis", interpolation="none")
ax.set_xlabel(r"time")
ax.set_ylabel(r"$u$ (pA)")

# plot width of u distribution
ax = fig.add_subplot(grid[2, 0])
u_width = res["u_width"]
ax.plot(time[:stop], u_width[:stop], color="darkorange")
ax.set_ylabel(r"$w$ (pA)")

# plot KLD between Lorentzian and recovery variable distribution
ax = fig.add_subplot(grid[3, 0])
kld = res["kld"]
ax.plot(time[:stop], kld[:stop], color="darkred")
ax.set_xlabel("time (ms)")
ax.set_ylabel("KLD")

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/u_dist.pdf')
plt.show()
