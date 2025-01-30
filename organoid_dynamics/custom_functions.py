import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def organoid_processing(well: int, age: int, spikes_well: np.ndarray, time: np.ndarray, time_ds: np.ndarray,
                        sigma: float, sigma2: float, width: float, width2: float, prominence: float, prominence2: float,
                        height: float, tau: int, tau2: int, return_spikes: bool = False) -> dict:

    print(f"    ... processing organoid {well}", flush=True)
    results = {"organoid": well, "age": age}

    # extract spikes and calculate firing rate statistics
    spikes = extract_spikes(time, time_ds, spikes_well)
    fr = np.mean(spikes, axis=0)
    results["rate_avg"] = 1e3*np.mean(fr)
    results["rate_het"] = 1e3*np.mean(np.std(spikes, axis=0))

    # calculate global dimensionality
    results["dim"] = get_dim(spikes, center=True)

    # calculate global inter-spike interval statistics
    isi = inter_spike_intervals(spikes)
    ff = fano_factor(spikes, tau=tau)
    results["spike_reg"] = np.mean(isi)
    results["ff"] = np.mean(ff)

    # calculate bursting frequency
    fr_smoothed = gaussian_filter1d(fr, sigma=sigma)
    fr_peaks, peak_props = find_peaks(fr_smoothed, width=width, prominence=np.max(fr_smoothed)*prominence,
                                      rel_height=height)
    ibi = np.diff(fr_peaks)
    results["burst_freq"] = np.mean(len(fr_peaks)/time_ds[-1])
    results["burst_reg"] = np.std(ibi) / np.mean(ibi) if len(fr_peaks) > 2 else 0

    # calculate intra-burst spiking statistics
    intraburst_results = {"rate_avg": [], "rate_het": [], "spike_reg": [], "ff": [], "dim": [], "freq": []}
    for left, right in zip(peak_props["left_ips"], peak_props["right_ips"]):
        spikes_tmp = spikes[:, int(left):int(right)]
        isi_tmp = inter_spike_intervals(spikes_tmp)
        ff_tmp = fano_factor(spikes_tmp, tau=tau2)
        fr_tmp = np.mean(spikes_tmp, axis=0)
        frs_tmp = gaussian_filter1d(fr_tmp, sigma=sigma2)
        peaks_tmp, _ = find_peaks(frs_tmp, prominence=np.max(frs_tmp)*prominence2, width=width2)
        intraburst_results["dim"].append(get_dim(spikes_tmp, center=True))
        intraburst_results["rate_avg"].append(1e3*np.mean(fr_tmp))
        intraburst_results["rate_het"].append(1e3*np.mean(np.std(spikes_tmp, axis=0)))
        intraburst_results["spike_reg"].append(np.mean(isi_tmp))
        intraburst_results["ff"].append(np.mean(ff_tmp))
        intraburst_results["freq"].append(len(peaks_tmp)/(time_ds[int(right)]-time_ds[int(left)]))
    for key, val in intraburst_results.items():
        results[f"intraburst_{key}"] = np.nanmean(val)

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
        results["spikes"] = spikes
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
