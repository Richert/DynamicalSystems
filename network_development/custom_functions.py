import numpy as np


def extract_spikes(time: np.ndarray, time_ds: np.ndarray, spikes: np.ndarray) -> np.ndarray:
    return np.asarray([_get_channel_spikes(time, time_ds, s) for s in spikes])


def fano_factor(spikes: np.ndarray, tau: int) -> np.ndarray:
    spike_counts = []
    idx = 0
    while idx < spikes.shape[1]:
        spike_counts.append(np.sum(spikes[:, idx:idx+tau], axis=1))
        idx += tau
    avg_spikes = np.mean(spike_counts, axis=0)
    ff = [np.var(s) / s_mean if s_mean > 0 else 0.0 for s, s_mean in zip(spike_counts, avg_spikes)]
    return np.asarray(ff)


def get_cov(x: np.ndarray, center: bool = True, alpha: float = 0.0) -> np.ndarray:
    if center:
        x_tmp = np.zeros_like(x)
        for i in range(x.shape[0]):
            x_tmp[i, :] = x[i, :] - np.mean(x[i, :])
    else:
        x_tmp = x
    return x_tmp @ x_tmp.T + alpha*np.eye(x_tmp.shape[0])


def get_dim(x: np.ndarray, center: bool = True, alpha: float = 0.0) -> float:
    cov = get_cov(x, center, alpha=alpha)
    eigs = np.real(np.abs(np.linalg.eigvals(np.asarray(cov, dtype=np.complex64))))
    return np.sum(eigs) ** 2 / np.sum(eigs ** 2)


def inter_spike_intervals(spikes: np.ndarray) -> np.ndarray:
    intervals = np.zeros((spikes.shape[0],))
    for i in range(spikes.shape[0]):
        spike_times = np.argwhere(spikes[i, :] > 0)[:, 0]
        if len(spike_times) > 2:
            spike_time_diff = spike_times[1:] - spike_times[:-1]
            intervals[i] = np.std(spike_time_diff) / np.mean(spike_time_diff)
    return intervals


def _get_channel_spikes(time: np.ndarray, time_ds: np.ndarray, spikes: np.ndarray):
    spike_times = np.zeros_like(time_ds)
    if spikes is not None and np.sum(spikes) > 0:
        s = np.asarray(spikes, dtype=np.int32) if spikes.shape else np.asarray([spikes], dtype=np.int32)
        spike_times = _get_spike_times(spike_times, s, time, time_ds)
    return spike_times


def _get_spike_times(spike_times: np.ndarray, s: np.ndarray, time: np.ndarray, time_ds: np.ndarray) -> np.ndarray:
    for spike in s:
        try:
            idx = np.argmin(np.abs(time_ds - time[spike]))
            spike_times[idx] = 1.0
        except IndexError:
            pass
    return spike_times
