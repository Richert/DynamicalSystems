import numpy as np
from scipy.stats import norm, cauchy


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
