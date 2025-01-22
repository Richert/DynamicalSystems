import numpy as np
from numba import njit, prange


def extract_spikes(n_channels: int, time: np.ndarray, spikes: np.ndarray) -> np.ndarray:
    spikes_lfp = np.zeros((n_channels, time.shape[0]))
    for i in range(n_channels):
        spike_times = np.zeros_like(time)
        s_tmp = spikes[i]
        if s_tmp is not None and np.sum(s_tmp) > 0:
            s = np.asarray(s_tmp, dtype=np.int32) if s_tmp.shape else np.asarray([s_tmp], dtype=np.int32)
            spike_times = _get_spike_times(spike_times, s, time)
        spikes_lfp[i, :] = spike_times
    return np.asarray(spikes_lfp)


def fano_factor(spikes: np.ndarray, tau: int) -> np.ndarray:
    spike_counts = []
    idx = 0
    while idx < spikes.shape[1]:
        spike_counts.append(np.sum(spikes[:, idx:idx+tau], axis=1))
        idx += tau
    return np.var(spike_counts, axis=0) / np.mean(spike_counts, axis=0)


@njit
def get_cov(x: np.ndarray, center: bool = True, alpha: float = 0.0) -> np.ndarray:
    if center:
        x_tmp = np.zeros_like(x)
        for i in range(x.shape[0]):
            x_tmp[i, :] = x[i, :] - np.mean(x[i, :])
    else:
        x_tmp = x
    return x_tmp @ x_tmp.T + alpha*np.eye(x_tmp.shape[0])


@njit
def get_dim(x: np.ndarray, center: bool = True, alpha: float = 0.0) -> float:
    cov = get_cov(x, center, alpha=alpha)
    eigs = np.abs(np.linalg.eigvals(cov))
    return np.sum(eigs) ** 2 / np.sum(eigs ** 2)


@njit
def inter_spike_intervals(spikes: np.ndarray) -> np.ndarray:
    intervals = []
    for i in range(spikes.shape[0]):
        spike_times = np.argwhere(spikes[i, :] > 0)[:, 0]
        if len(spike_times) > 2:
            spike_time_diff = spike_times[1:] - spike_times[:-1]
            intervals.append(np.std(spike_time_diff) / np.mean(spike_time_diff))
    return np.asarray(intervals)

@njit
def _get_spike_times(spike_times: np.ndarray, s: np.ndarray, time: np.ndarray) -> np.ndarray:
    for spike in s:
        idx = np.argmin(np.abs(time - time[spike]))
        spike_times[idx] = 1.0
    return spike_times
