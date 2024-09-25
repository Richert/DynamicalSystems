import numpy as np
from numpy.distutils.system_info import flame_info
from scipy.stats import norm, cauchy
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares, curve_fit
from scipy.signal import find_peaks
from typing import Union, Iterable, Callable


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def gaussian(n, mu: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = norm.rvs(loc=mu, scale=delta)
        while s <= lb or s >= ub:
            s = norm.rvs(loc=mu, scale=delta)
        samples[i] = s
    return samples


def fano_factor(spikes: list, max_time: int, tau: int) -> np.ndarray:
    idx = 0
    ff = []
    while idx < max_time:
        spike_counts = []
        for s in spikes:
            spike_counts.append(np.sum((s >= idx) * (s < idx + tau)))
        ff.append(np.var(spike_counts)/np.mean(spike_counts))
        idx += tau
    return np.asarray(ff)


def fano_factor2(spikes: list, max_time: int, tau: int) -> np.ndarray:
    ff = []
    for s in spikes:
        spike_counts = []
        idx = 0
        while idx < max_time:
            spike_counts.append(np.sum((s >= idx) * (s < idx + tau)))
            idx += tau
        ff.append(np.var(spike_counts) / np.mean(spike_counts))
    return np.asarray(ff)


def convolve_exp(x: np.ndarray, tau: float, dt: float) -> np.ndarray:
    y = np.zeros((x.shape[1],))
    ys = []
    for step in range(x.shape[0]):
        y = y + dt*(-y/tau) + x[step, :]
        ys.append(y)
    return np.asarray(ys)


def _sequentiality(events: list, max_lag: int = 100, n_bins: int = 100) -> list:
    N = len(events)
    bins = np.linspace(-max_lag, max_lag, num=n_bins)
    s = []
    for n1 in range(N):
        event_entropy = []
        for e in events[n1]:
            diffs = []
            for n2 in range(N):
                e2 = events[n2]
                if n1 != n2 and len(e2) > 0:
                    min_idx = np.argmin(np.abs(e2 - e))
                    diff = e - e2[min_idx]
                    if np.abs(diff) < max_lag:
                        diffs.append(diff)
            sh = entropy(diffs, bins=bins)
            event_entropy.append(sh if np.isfinite(sh) else 0.0)
        s.append(np.mean(event_entropy) if len(event_entropy) > 0 else 0.0)
    return s


def sequentiality(events: list, neighborhood: int, overlap: float = 0.5, time_window: int = 100, n_bins: int = 100,
                  method: str = "custom") -> list:
    N = len(events)
    seq = []
    idx = 0
    while idx < N:

        # select part of network that we will calculate the sequentiality for
        start = np.maximum(0, idx-int(overlap*neighborhood))
        stop = np.minimum(N, start+neighborhood)
        idx = stop

        # calculate sequentiality
        if method == "sqi":
            pe, ts = sqi(events[start:stop], time_window=time_window, n_bins=n_bins)
            s = np.sqrt(pe*ts)
        else:
            s = _sequentiality(events[start:stop], max_lag=time_window, n_bins=n_bins)
        seq.extend(s)

    return seq


def dist(x: int, method: str = "inverse", zero_val: float = 1.0, sigma: float = 1.0) -> float:
    if x == 0:
        return zero_val
    if method == "inverse":
        return np.abs(x)**(-1/sigma)
    if method == "exp":
        return np.exp(-np.abs(x)/sigma)
    if method == "gaussian":
        return norm.pdf(x, scale=sigma)
    raise ValueError("Invalid method.")


def entropy(x: list, bins: np.ndarray) -> float:
    counts, _ = np.histogram(x, bins=bins)
    idx = counts > 0
    ps = counts[idx] / np.sum(counts[idx])
    uniform = np.ones_like(counts) / counts.shape[0]
    return np.sum(ps * np.log(ps)) / np.sum(uniform * np.log(uniform))


def sqi(events: list, time_window: int = 100, n_bins: int = 100) -> tuple:

    # preparations
    N = len(events)
    step = np.round(time_window/n_bins, decimals=0)
    bins = np.arange(0, time_window, step=step, dtype=np.int32)
    max_time = np.max([e[-1] for e in events if len(e) > 0])
    spikes = np.zeros((N, max_time + 1))
    for i, e in enumerate(events):
        spikes[i, e] = 1.0

    # calculations
    peak_entropy, temporal_sparsity = [], []
    tbin = bins[0]
    while tbin < max_time:

        ps, rate_entropy = [], []
        for ub in bins[1:]:
            summed_spikes = np.sum(spikes[:, tbin:tbin+ub])
            ps.append(summed_spikes/N)
            if summed_spikes > 0:
                neuron_rates = np.sum(spikes[:, tbin:tbin+ub], axis=1) / summed_spikes
                idx = neuron_rates > 0
                rate_entropy.append(-np.sum(neuron_rates[idx] * np.log(neuron_rates[idx])) / np.log(N))
            tbin += ub

        ps = np.asarray(ps)
        idx = ps > 0
        peak_entropy.append(-np.sum(ps[idx] * np.log(ps[idx])) / np.log(n_bins))
        temporal_sparsity.append(1 - np.mean(rate_entropy))

    pe, ts = np.asarray(peak_entropy), np.asarray(temporal_sparsity)
    return pe, ts


def get_dim(x: np.ndarray) -> float:
    cov = x.T @ x
    eigs = np.abs(np.linalg.eigvals(cov))
    return np.sum(eigs) ** 2 / np.sum(eigs ** 2)


def dimensionality_filtered(x: np.ndarray, sigmas: list, windows: list) -> dict:

    N = x.shape[1]
    ds = {s: [] for s in sigmas}
    for s in sigmas:

        # apply smoothing
        if s > 0:
            x_tmp = np.zeros_like(x)
            for i in range(N):
                x_tmp[:, i] = gaussian_filter1d(x[:, i], sigma=s)
        else:
            x_tmp = x[:, :]

        # calculate time-window specific dimensionality
        for w in windows:

            idx = 0
            d_col = []
            while idx + w <= x_tmp.shape[0]:

                x_tmp2 = x_tmp[idx:idx+w, :]
                dim = get_dim(x_tmp2)
                d_col.append(dim)
                idx += w

            ds[s].append(np.mean(d_col))

    return ds


def impulse_response_fit(x: np.ndarray, time: np.ndarray, f: Callable, bounds: tuple, **kwargs) -> tuple:

    # normalize signal
    x -= np.min(x)
    x /= np.max(x)

    # fit
    p = curve_fit(f, time, x, bounds=bounds, **kwargs)

    # prediction
    y = f(time, *tuple(p[0]))

    return p[0], y


def get_peaks(x: np.ndarray, prominence: float = 0.5, epsilon: float = 1e-3, **kwargs):
    peaks, _ = find_peaks(x, prominence=prominence, **kwargs)
    if len(peaks) == 1:
        idx = np.argwhere(x[peaks[0]:] < x[peaks[0]]-epsilon).squeeze()[0]
        peaks = np.arange(peaks[0], idx).tolist()
    return peaks


def exponential_kernel(t: Union[float, np.ndarray], offset: float, delay: float, scale: float, tau: float):
    """Mono-exponential kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param offset: Constant offset of the kernel. Must be a scalar.
    :param delay: Delay until onset of the kernel response. Must be a scalar.
    :param scale: Scaling of the kernel function. Must be a scalar.
    :param tau: Decay time constant of the kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    t1 = t - delay
    on = 1.0 * (t1 > 0.0)
    return offset + on * scale * np.exp(-t1 / tau)


def alpha(t: Union[float, np.ndarray], offset: float, delay: float, scale: float, tau: float):
    """Mono-exponential kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param offset: Constant offset of the kernel. Must be a scalar.
    :param delay: Delay until onset of the kernel response. Must be a scalar.
    :param scale: Scaling of the kernel function. Must be a scalar.
    :param tau: Decay time constant of the kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    t1 = t - delay
    on = 1.0 * (t1 > 0.0)
    return offset + on * scale * t1 * np.exp(1 - t1 / tau) / tau


def biexponential(t: Union[float, np.ndarray], offset: float, delay: float, scale: float, tau_r: float, tau_d: float):
    """Mono-exponential kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param offset: Constant offset of the kernel. Must be a scalar.
    :param delay: Delay until onset of the kernel response. Must be a scalar.
    :param scale: Scaling of the kernel function. Must be a scalar.
    :param tau_r: Rise time constant of the kernel. Must be a scalar.
    :param tau_d: Decay time constant of the kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    t1 = t - delay
    on = 1.0 * (t1 > 0.0)
    return offset + on * scale * tau_d * tau_r * (np.exp(-t1 / tau_d) - np.exp(-t1 / tau_r)) / (tau_d - tau_r)
