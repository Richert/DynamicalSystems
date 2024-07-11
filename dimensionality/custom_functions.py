import numpy as np
from scipy.stats import norm, cauchy
from numba import njit, prange, config
config.THREADING_LAYER = 'tbb'


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


#@njit(fastmath=True, parallel=True)
def _sequentiality(events: list) -> list:
    N = len(events)
    s = []
    for n1 in range(N):
        event_entropy = []
        for e in events[n1]:
            diffs = []
            for n2 in prange(N):
                e2 = events[n2]
                if n1 != n2 and len(e2) > 0:
                    min_idx = np.argmin(np.abs(e2 - e))
                    diffs.append(e - e2[min_idx])
            event_entropy.append(entropy(diffs))
        s.append(np.mean(np.asarray(event_entropy)))
    return s


def sequentiality(events: list, neighborhood: int, overlap: float = 0.5) -> list:
    N = len(events)
    seq = []
    idx = 0
    while idx < N:

        # select part of network that we will calculate the sequentiality for
        start = np.maximum(0, idx-int(overlap*neighborhood))
        stop = np.minimum(N, start+neighborhood)
        idx = stop

        # calculate sequentiality
        seq.extend(_sequentiality(events[start:stop]))

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


#@njit()
def entropy(x: list) -> float:
    x = np.asarray(x)
    values = np.unique(x)
    counts = np.asarray([np.sum(x == v) for v in values])
    counts = counts / np.sum(counts)
    uniform = np.ones_like(counts) / counts.shape[0]
    return np.sum(counts * np.log(counts)) / np.sum(uniform * np.log(uniform))
