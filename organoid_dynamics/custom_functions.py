import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, welch
from equidistantpoints import EquidistantPoints
from scipy.stats import norm, cauchy


def organoid_lfp_analysis(well: int, age: int, lfps: np.ndarray, time_ds: np.ndarray, nperseg: int, fmax: float) -> dict:

    print(f"    ... processing LFPs of organoid {well}", flush=True)
    results = {"organoid": well, "age": age}

    # preprocess LFP array
    lfps = lfps * 1e6
    if len(lfps) == 1:
        lfps = lfps[0]
    if lfps.shape[0] > lfps.shape[1]:
        lfps = lfps.T

    # calculate average potential and potential fluctuations
    lfp_avg = np.mean(lfps, axis=0)
    results["lfp_avg"] = np.mean(lfp_avg)
    results["lfp_var"] = np.std(lfp_avg)

    # calculate global dimensionality
    results["dim"] = get_dim(lfps, center=True)

    # calculate spectral properties
    freqs, power = get_psd(lfp_avg, fs=float(1.0/(time_ds[1]-time_ds[0])), nperseg=nperseg, fmax=fmax, detrend=True)

    results["psd"] = power
    results["freq"] = freqs
    results["max_freq"] = freqs[np.argmax(power)]
    results["max_pow"] = np.max(power)

    return results


def organoid_spike_analysis(well: int, age: int, spikes_well: np.ndarray, time: np.ndarray, time_ds: np.ndarray,
                            sigma: float, width: float, width2: float, prominence: float, prominence2: float,
                            height: float, tau: float, return_spikes: bool = False) -> dict:

    print(f"    ... processing spikes of organoid {well}", flush=True)
    results = {"organoid": well, "age": age}
    intraburst_results = {"rate_avg": [], "rate_het": [], "spike_reg": [], "dim": [], "freq": []}

    # extract spikes and calculate firing rate statistics
    spikes = extract_spikes(time, time_ds, spikes_well)
    spikes_smoothed = convolve_exp(spikes, tau=tau, dt=float(time_ds[1]-time_ds[0]))
    fr = np.mean(spikes_smoothed, axis=0)
    results["rate_avg"] = 1e3 * np.mean(fr)
    results["rate_het"] = 1e3 * np.mean(np.std(spikes, axis=0))

    # calculate global dimensionality
    results["dim"] = get_dim(spikes_smoothed, center=True)

    # calculate global inter-spike interval statistics
    isi = inter_spike_intervals(spikes)
    results["spike_reg"] = np.mean(isi)

    # calculate bursting frequency
    fr_smoothed = gaussian_filter1d(fr, sigma=sigma)
    fr_peaks, peak_props = find_peaks(fr_smoothed, width=width, prominence=np.max(fr_smoothed) * prominence,
                                      rel_height=height)
    results["burst_freq"] = len(fr_peaks) / time_ds[-1]

    if len(fr_peaks) > 2:

        ibi = np.diff(fr_peaks)
        results["burst_reg"] = np.std(ibi) / np.mean(ibi)

        # calculate intra-burst spiking statistics
        for left, right in zip(peak_props["left_ips"], peak_props["right_ips"]):
            spikes_tmp = spikes[:, int(left):int(right)]
            spikes_smooth_tmp = spikes_smoothed[:, int(left):int(right)]
            isi_tmp = inter_spike_intervals(spikes_tmp)
            fr_tmp = np.mean(spikes_smooth_tmp, axis=0)
            peaks_tmp, _ = find_peaks(fr_tmp, prominence=np.max(fr_tmp) * prominence2, width=width2)
            intraburst_results["dim"].append(get_dim(spikes_smooth_tmp, center=True))
            intraburst_results["rate_avg"].append(1e3 * np.mean(fr_tmp))
            intraburst_results["rate_het"].append(1e3 * np.mean(np.std(spikes_tmp, axis=0)))
            intraburst_results["spike_reg"].append(np.mean(isi_tmp))
            intraburst_results["freq"].append(len(peaks_tmp) / (time_ds[int(right)] - time_ds[int(left)]))
        for key, val in intraburst_results.items():
            results[f"intraburst_{key}"] = np.nanmean(val)

    else:

        results["burst_reg"] = 0.0
        for key in intraburst_results:
            results[f"intraburst_{key}"] = 0.0

    # # plotting: single organoid and time point
    # fig, axes = plt.subplots(nrows=3, figsize=(12, 9))
    # fig.suptitle(f"Organoid ID: {well}, Age: {age} days")
    # ax = axes[0]
    # ax.plot(time_ds, fr, label="raw")
    # ax.plot(time_ds, fr_smoothed, label="smoothed")
    # for l, r in zip(peak_props["left_ips"], peak_props["right_ips"]):
    #     ax.axvline(x=time_ds[int(l)], color="black", linestyle="dashed")
    #     ax.axvline(x=time_ds[int(r)], color="red", linestyle="dashed")
    # ax.set_ylabel("rate")
    # ax.set_xlabel("time")
    # ax.legend()
    # ax = axes[1]
    # ax.imshow(spikes, aspect="auto", interpolation="none", cmap="Greys")
    # ax.set_ylabel("neurons")
    # ax.set_xlabel("steps")
    # ax = axes[2]
    # ax.hist(isi)
    # ax.set_xlabel("ISI")
    # ax.set_ylabel("count")
    # plt.tight_layout()
    # plt.show()

    if return_spikes:
        results["spikes"] = spikes_smoothed
    return results


def get_bursting_stats(x: np.ndarray, sigma: float, burst_width: float, rel_burst_height, width_at_height: float) -> dict:

    # smooth signal
    x = gaussian_filter1d(x, sigma=sigma)

    # extract signal peaks
    peaks, props = find_peaks(x, width=burst_width, prominence=np.max(x) * rel_burst_height, rel_height=width_at_height)
    ibi = np.diff(peaks)

    # extract bursting characteristics
    results = {}
    results["n_bursts"] = len(peaks)
    results["ibi_mean"] = np.mean(ibi)
    results["ibi_std"] = np.std(ibi) / np.mean(ibi)
    results["burst_width"] = np.mean(props["widths"])

    return results

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


def convolve_exp(x: np.ndarray, tau: float, dt: float, d: float = 0.0, normalize: bool = True) -> np.ndarray:
    y = np.zeros((x.shape[0],))
    ys = []
    delay = int(d/dt)
    for step in range(x.shape[1]):
        inp = x[:, step - delay] if step >= delay else 0.0
        y = y + dt*(-y/tau) + inp
        ys.append(y)
    ys = np.asarray(ys).T
    if normalize:
        for i in range(ys.shape[0]):
            y_max = np.max(ys[i, :])
            if y_max > 0:
                ys[i, :] /= y_max
    return ys


def random_connectivity(n: int, m: int, p: float, homogeneous_weights: bool = False, normalize: bool = False) -> np.ndarray:
    C = np.zeros((n, m))
    n_conns = int(m*p)
    positions = np.arange(start=0, stop=m)
    for row in range(n):
        cols = np.random.permutation(positions)[:n_conns]
        C[row, cols] = 1.0 if homogeneous_weights else np.random.rand()
        if normalize:
            C[row, :] /= np.sum(C[row, :])
    return C


def circular_connectivity(n1: int, n2: int, p: float, homogeneous_weights: bool = False, dist: str = "gaussian",
                          scale: float = 1.0, normalize: bool = False) -> np.ndarray:
    p1 = np.linspace(-np.pi, np.pi, n1)
    p2 = np.linspace(-np.pi, np.pi, n2)
    n_conn = int(p*n2)
    W = np.zeros((n1, n2))
    f = gaussian if dist == "gaussian" else lorentzian
    for i in range(n1):
        distances = np.asarray([circular_distance(p1[i], p2[j]) for j in range(n2)])
        d_samples = f(n_conn, loc=0.0, scale=scale, lb=-1.0, ub=1.0)
        indices = np.asarray([np.argmin(np.abs(distances - d)) for d in d_samples])
        indices_unique = np.unique(indices)
        if homogeneous_weights:
            W[i, indices_unique] = len(indices) / len(indices_unique)
        else:
            for idx in indices_unique:
                W[i, idx] = np.sum(indices == idx)
        if normalize:
            W[i, :] /= len(indices)
    return W


def spherical_connectivity(n1: int, n2: int, p: float, homogeneous_weights: bool = False, dist: str = "gaussian",
                          scale: float = 1.0, normalize: bool = False) -> np.ndarray:

    p1 = EquidistantPoints(n_points=n1)
    p1 = np.asarray(p1.ecef)
    p2 = EquidistantPoints(n_points=n2)
    p2 = np.asarray(p2.ecef)
    n_conn = int(p * n2)
    W = np.zeros((n1, n2))
    f = gaussian if dist == "gaussian" else lorentzian
    for i in range(n1):
        distances = np.asarray([spherical_distance(p1[i], p2[j]) for j in range(n2)])
        d_samples = f(n_conn, loc=0.0, scale=scale, lb=-1.0, ub=1.0)
        indices = np.asarray([np.argmin(np.abs(distances - d)) for d in d_samples])
        indices_unique = np.unique(indices)
        if homogeneous_weights:
            W[i, indices_unique] = len(indices) / len(indices_unique)
        else:
            for idx in indices_unique:
                W[i, idx] = np.sum(indices == idx)
        if normalize:
            W[i, :] /= len(indices)
    return W


def lorentzian(n: int, loc: float, scale: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=loc, scale=scale)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=loc, scale=scale)
        samples[i] = s
    return samples


def gaussian(n, loc: float, scale: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = norm.rvs(loc=loc, scale=scale)
        while s <= lb or s >= ub:
            s = norm.rvs(loc=loc, scale=scale)
        samples[i] = s
    return samples


def spherical_distance(p1: np.ndarray, p2: np.ndarray, epsilon=1e-10) -> float:
    if np.sum((p1 - p2)**2) < epsilon:
        return np.inf
    d = np.arccos(np.dot(p1, p2))
    return d


def circular_distance(p1: float, p2: float, epsilon=1e-10) -> float:
    if (p1 - p2)**2 < epsilon:
        return np.inf
    return np.sin(0.5*(p1 - p2))


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x - y)**2))


def get_psd(x: np.ndarray, fs: float, nperseg: int, fmax: float, detrend: bool = True, **kwargs) -> tuple:
    freqs, power_raw = welch(x, fs=fs, nperseg=nperseg, **kwargs)
    idx = freqs <= fmax
    power_log = np.log(power_raw[idx])
    if detrend:
        X = np.vstack([np.arange(len(power_log)), np.ones(len(power_log))]).T
        coefs = np.linalg.lstsq(X, power_log)[0]
        power_log -= X @ coefs
    return freqs[idx], power_log


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
