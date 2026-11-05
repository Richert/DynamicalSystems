"""Per-cell tuning of SPN firing to each behaviour feature, with the lag of maximal tuning structure,
per drug condition (both cohorts; SPN data files of build_spn_dataset.py).

Rate: spike events smoothed with a Gaussian (kernel_s), sampled every `step` frames. Features: the
11 AR-HMM features in the HMM's transforms. Bins: n_bins quantile bins of the feature over the
cohort's Control recordings (common edges for all conditions).
Per cell, feature and lag (feature shifted by -max_lag_s ... +max_lag_s; lag > 0: behaviour follows
the SPN):
  tuning curve   mean rate per bin (bins with >= min_bin_s of data)
  structure      eta^2 = variance of the rate explained by the bins (between-bin / total
                 variance; captures non-monotonic tuning)
Recordings are analysed in seg_s segments (Control: 1, Amphetamine: 3) and eta^2 averaged over
segments before the best lag is chosen (equal data per estimate). Significance: the best eta^2
exceeds the null_q percentile of the same statistic (max over lags) with the behaviour circularly
shifted by >= null_min_s (n_null shifts, pooled over the recording's cells).

Outputs (OUT_DIR): cells_<drug>.csv.gz (cell x feature: best lag, eta2, null threshold, tuned,
direction = Spearman sign of the tuning curve, tuning curve at the best lag), bins_<drug>.json.

Equal occupancy (EQUAL_BINS=1): per_bin_s of data drawn from every bin (n_draw draws), outputs
cells_<drug>_eq.csv.gz.

  conda run -n parker python analysis_spn_feature_tuning_cells.py [n_jobs]
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter1d

data_dir = "/home/rgast/data/parker_data/dataset"
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/spn_feature_tuning_cells")
drugs = ["Clozapine", "Haloperidol"]
fs = 20.0
kernel_s = 0.1
step = 2                                    # 10 Hz
max_lag_s, lag_step_s = 0.5, 0.05
n_bins, min_bin_s = 8, 5.0
seg_s = 900.0
n_null, null_min_s, null_q = 5, 60.0, 99
# equal occupancy (EQUAL_BINS=1): per bin the same time (per_bin_s) drawn at random, bins with less
# data dropped, eta^2 averaged over n_draw draws -> the chance level of eta^2 is the same in every
# condition and does not depend on how much the drug changes the feature's distribution
equal_bins = bool(int(os.environ.get("EQUAL_BINS", "0")))
per_bin_s, n_draw = 5.0, 5
speed_eps, turn_eps = 0.5, 5.0
FEATURES = {"speed": lambda b: np.log10(b["speed"][:] + speed_eps),
            "head_speed": lambda b: np.log10(b["head_speed"][:] + speed_eps),
            "turn": lambda b: np.arcsinh(b["turn"][:] / turn_eps),
            "head_turn": lambda b: np.arcsinh(b["head_turn"][:] / turn_eps),
            "head_angle": lambda b: b["head_angle"][:], "bend": lambda b: b["bend"][:],
            "body_len": lambda b: b["body_len"][:], "ear_width": lambda b: b["ear_width"][:],
            "hip_width": lambda b: b["hip_width"][:], "wall_dist": lambda b: b["wall_dist"][:],
            "wall_ahead": lambda b: b["wall_ahead"][:]}
seed = 1

def conditions(drug):
    return ["Control", "Amphetamine", f"{drug} high", f"A + {drug} high"]

def control_bins(drug):
    vals = {f: [] for f in FEATURES}
    with h5py.File(f"{data_dir}/{drug.lower()}_spn.h5", "r") as h:
        for d in h:
            r = h[d]["rec0"]
            if r.attrs["condition"] != "Control" or not r["behaviour"].attrs.get("available", False):
                continue
            for f, fn in FEATURES.items():
                v = fn(r["behaviour"])
                vals[f].append(v[np.isfinite(v)])
    return {f: np.quantile(np.concatenate(v), np.linspace(0, 1, n_bins + 1)).tolist() for f, v in vals.items()}

def eta2_all(R, idx, valid_bins):
    """eta^2 per cell (rows of R, mean-centred) for bin indices idx (-1: excluded)."""
    ok = idx >= 0
    Rk, ik = R[:, ok], idx[ok]
    B = np.zeros((len(ik), n_bins), np.float32)
    B[np.arange(len(ik)), ik] = 1
    cnt = B.sum(0)
    use = valid_bins & (cnt > 0)
    S = Rk @ B
    m = Rk.mean(1, keepdims=True)
    means = S / np.maximum(cnt, 1)
    between = ((means[:, use] - m) ** 2 * cnt[use]).sum(1)
    total = ((Rk - m) ** 2).sum(1)
    return between / np.maximum(total, 1e-12), means

def equalise(idx, rng):
    """Bin indices with per_bin_s of data per bin drawn at random (other samples -1); bins with
    less data are dropped."""
    k = int(per_bin_s * fs / step)
    out = np.full_like(idx, -1)
    for b_ in range(n_bins):
        pos = np.flatnonzero(idx == b_)
        if len(pos) >= k:
            out[rng.choice(pos, k, replace=False)] = b_
    return out

def eta_eq(R, idx, rng):
    """eta^2 (and bin means) averaged over n_draw equal-occupancy draws."""
    e_sum, m_sum, nb = 0, 0, 0
    for _ in range(n_draw):
        ie = equalise(idx, rng)
        valid = np.bincount(ie[ie >= 0], minlength=n_bins) > 0
        e, m = eta2_all(R, ie, valid)
        e_sum, m_sum, nb = e_sum + e / n_draw, m_sum + m / n_draw, valid.sum()
    return e_sum, m_sum, nb

def recording(drug, day, rec, bins):
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(abs(hash((day, rec, seed))) % 2 ** 32)
    with h5py.File(f"{data_dir}/{drug.lower()}_spn.h5", "r") as h:
        g, r = h[day], h[day][rec]
        cond, mouse, spn = r.attrs["condition"], g.attrs["mouse"], g.attrs["spn"]
        feats = {f: fn(r["behaviour"]) for f, fn in FEATURES.items()}
        ptr, fr = r["event_ptr"][:], r["event_frames"][:]
        T = r["C"].shape[1]
    n = len(ptr) - 1
    E = np.zeros((n, T), np.float32)
    for i in range(n):
        E[i, fr[ptr[i]:ptr[i + 1]]] = 1
    Rfull = gaussian_filter1d(E, kernel_s * fs, axis=1) * fs          # Hz
    lags = np.arange(-int(max_lag_s * fs), int(max_lag_s * fs) + 1, int(lag_step_s * fs))
    L = int(seg_s * fs)
    segs = [(a, min(a + L, T)) for a in range(0, max(1, T // L) * L, L)]
    pad = lags.max()
    rows = []
    for f, x in feats.items():
        edges = np.asarray(bins[f])
        xb = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
        xb[~np.isfinite(x)] = -1
        eta = np.zeros((n, len(lags)))
        curves = np.zeros((n, len(lags), n_bins))
        null_max = []
        nbins_valid = np.full(len(lags), n_bins)
        for a, b in segs:
            t = np.arange(a + pad, b - pad, step)
            R = Rfull[:, t]
            for j, lag in enumerate(lags):
                idx = xb[t + lag]                                   # behaviour lag frames later
                if equal_bins:
                    e, mns, nbv = eta_eq(R, idx, rng)
                    nbins_valid[j] = nbv
                else:
                    cnt = np.bincount(idx[idx >= 0], minlength=n_bins) * step / fs
                    e, mns = eta2_all(R, idx, cnt >= min_bin_s)      # bins with >= min_bin_s of data
                eta[:, j] += e / len(segs)
                curves[:, j] += mns / len(segs)
            for _ in range(n_null):
                s = rng.integers(int(null_min_s * fs), max(int(null_min_s * fs) + 1, (b - a) - int(null_min_s * fs)))
                xs = np.roll(xb[a:b], s)
                best = np.zeros(n)
                for lag in lags:
                    idx = xs[t - a + lag]
                    if equal_bins:
                        best = np.maximum(best, eta_eq(R, idx, rng)[0])
                    else:
                        cnt = np.bincount(idx[idx >= 0], minlength=n_bins) * step / fs
                        best = np.maximum(best, eta2_all(R, idx, cnt >= min_bin_s)[0])
                null_max.append(best)
        thr = np.percentile(np.concatenate(null_max), null_q)
        jb = eta.argmax(1)
        cb = curves[np.arange(n), jb]
        from scipy.stats import spearmanr
        direction = np.array([np.sign(spearmanr(np.arange(n_bins), c).correlation) if np.std(c) > 0 else 0
                              for c in cb])
        rows.append(pd.DataFrame(dict(drug=drug, day=day, rec=rec, condition=cond, mouse=mouse, spn=spn,
                                      cell=np.arange(n), feature=f, lag=lags[jb] / fs, eta2=eta.max(1),
                                      thr=thr, tuned=eta.max(1) > thr, direction=direction,
                                      mean_rate=E.mean(1) * fs, n_bins_used=int(nbins_valid[jb].min()) if len(jb) else n_bins,
                                      **{f"c{k}": cb[:, k] for k in range(n_bins)})))
    print(drug, day, rec, flush=True)
    return pd.concat(rows, ignore_index=True)

def main(n_jobs=4):
    from joblib import Parallel, delayed
    os.makedirs(out_dir, exist_ok=True)
    for drug in drugs:
        bins = control_bins(drug)
        json.dump(bins, open(f"{out_dir}/bins_{drug.lower()}.json", "w"))
        with h5py.File(f"{data_dir}/{drug.lower()}_spn.h5", "r") as h:
            todo = [(d, r) for d in h for r in ("rec0", "rec1")
                    if h[d][r].attrs["condition"] in conditions(drug)
                    and h[d][r]["behaviour"].attrs.get("available", False)]
        if os.environ.get("ONLY"):
            todo = todo[:int(os.environ["ONLY"])]
        res = Parallel(n_jobs=n_jobs)(delayed(recording)(drug, d, r, bins) for d, r in todo)
        cells = pd.concat(res, ignore_index=True)
        cells.to_csv(f"{out_dir}/cells_{drug.lower()}{'_eq' if equal_bins else ''}.csv.gz", index=False)
        pd.set_option("display.width", 220)
        print(cells.groupby(["spn", "condition", "feature"]).tuned.mean().unstack().round(2).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
