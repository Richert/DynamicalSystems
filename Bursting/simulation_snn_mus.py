import numpy as np
from rectipy import Network
from scipy.stats import cauchy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
from numba import njit
plt.rcParams['backend'] = 'TkAgg'


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


# define parameters
###################

# define worker pool
pool = Parallel(n_jobs=10)

# define level of heterogeneity
delta = 2.0

# define conditions
cond_map = {
        "no_sfa_1": {"kappa": 0.0, "eta": 0.0, "eta_inc": 30.0, "eta_init": -30.0, "b": -5.0, "delta": delta},
        "weak_sfa_1": {"kappa": 100.0, "eta": 0.0, "eta_inc": 35.0, "eta_init": 0.0, "b": -5.0, "delta": delta},
        "strong_sfa_1": {"kappa": 300.0, "eta": 0.0, "eta_inc": 50.0, "eta_init": 0.0, "b": -5.0, "delta": delta},
        "no_sfa_2": {"kappa": 0.0, "eta": -150.0, "eta_inc": 190.0, "eta_init": -50.0, "b": -20.0, "delta": delta},
        "weak_sfa_2": {"kappa": 100.0, "eta": -20.0, "eta_inc": 70.0, "eta_init": -100.0, "b": -20.0, "delta": delta},
        "strong_sfa_2": {"kappa": 300.0, "eta": 40.0, "eta_inc": 100.0, "eta_init": 0.0, "b": -20.0, "delta": delta},
    }

# condition
conditions = ["no_sfa_1", "no_sfa_2", "weak_sfa_1", "weak_sfa_2", "strong_sfa_1", "strong_sfa_2"]
for cond in conditions:

    # model parameters
    N = 2000
    C = 100.0   # unit: pF
    k = 0.7  # unit: None
    v_r = -60.0  # unit: mV
    v_t = -40.0  # unit: mV
    eta = 0.0  # unit: pA
    mu = 0.0  # unit: pA
    Delta = cond_map[cond]["delta"]
    kappa = cond_map[cond]["kappa"]
    tau_u = 50.0
    b = cond_map[cond]["b"]
    tau_s = 6.0
    tau_x = 300.0
    g = 15.0
    E_r = 0.0

    v_reset = -1000.0
    v_peak = 1000.0

    # define inputs
    T = 7000.0
    dt = 1e-2
    dts = 2e-1
    cutoff = 1000.0
    inp = np.zeros((int(T/dt), 1)) + cond_map[cond]["eta"]
    inp[:int(300.0/dt), 0] += cond_map[cond]["eta_init"]
    inp[int(2000/dt):int(5000/dt), 0] += cond_map[cond]["eta_inc"]

    # define lorentzian distribution of etas
    mus = mu + Delta * np.tan(0.5*np.pi*(2*np.arange(1, N+1)-N-1)/(N+1))

    # define connectivity
    # W = random_connectivity(N, N, 0.2)

    # run the model
    ###############

    # initialize model
    node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": v_t, "eta": eta, "tau_u": tau_u, "b": b, "kappa": kappa,
                 "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_r, "tau_x": tau_x, "mu": mus}

    # initialize model
    net = Network(dt=dt, device="cpu")
    net.add_diffeq_node("sfa", f"config/snn/adik_mu", #weights=W, source_var="s", target_var="s_in",
                        input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                        node_vars=node_vars.copy(), op="adik_op_mu", spike_reset=v_reset, spike_threshold=v_peak,
                        verbose=False, clear=True, N=N, float_precision="float64")

    # perform simulation
    obs = net.run(inputs=inp, sampling_steps=int(dts/dt), verbose=True, cutoff=int(cutoff/dt),
                  record_vars=[("sfa", "u", False), ("sfa", "v", False), ("sfa", "x", False)], enable_grad=False)
    s, v, u, x = (obs.to_dataframe("out"), obs.to_dataframe(("sfa", "v")), obs.to_dataframe(("sfa", "u")),
                  obs.to_dataframe(("sfa", "x")))
    del obs
    time = s.index

    # calculate the mean-field quantities
    spikes = s.values
    r = np.mean(spikes, axis=1) / tau_s
    u_widths, u_errors = get_fwhm(u.values, pool=pool, plot_steps=100000, n_bins=500, eval_range=100.0,
                                  tol=1e-3, options={"maxiter": 500}, min_width=1e-10, method="Nelder-Mead")
    u = np.mean(u.values, axis=1)
    v = np.mean(v.values, axis=1)
    s = np.mean(s.values, axis=1)
    x = np.mean(x.values, axis=1)

    # calculate the kuramoto order parameter
    ko_y = v - v_r
    ko_x = np.pi*C*r/k
    z = (1 - ko_x + 1.0j*ko_y)/(1 + ko_x - 1.0j*ko_y)

    # save results to file
    results = {"spikes": spikes, "v": v, "u": u, "x": x, "r": r, "s": s, "z": 1 - np.abs(z), "theta": np.imag(z),
               "u_width": u_widths, "u_errors": u_errors}
    pickle.dump({"results": results, "params": node_vars}, open(f"results/snn_bs_{cond}.pkl", "wb"))

    # plot results
    fig, ax = plt.subplots(nrows=2, figsize=(12, 6))
    ax[0].imshow(spikes.T, interpolation="none", cmap="Greys", aspect="auto")
    ax[0].set_ylabel(r'neuron id')
    ax[1].plot(time, r)
    ax[1].set_ylabel(r'$r(t)$')
    ax[1].set_xlabel('time')
    plt.tight_layout()

    # plot distribution dynamics
    fig2, ax = plt.subplots(nrows=3, figsize=(12, 6))
    ax[0].plot(time, v, color="royalblue")
    ax[0].set_title("v (mV)")
    ax[1].plot(time, u, color="darkorange")
    ax[1].fill_between(time, u - u_widths, u + u_widths, alpha=0.3, color="darkorange", linewidth=0.0)
    ax[1].set_title("u (pA)")
    ax[2].plot(time, u_errors, color="red")
    ax[2].set_title("KLD(u)")
    ax[2].set_xlabel("time (ms)")
    fig2.suptitle("SNN")
    plt.tight_layout()
    plt.show()
