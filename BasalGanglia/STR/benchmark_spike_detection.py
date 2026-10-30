"""Benchmark of spike-event detection on OASIS output against GCaMP7f ground truth.

Ground truth: CASCADE data set DS29 (jGCaMP7f, mouse V1, 2p imaging with simultaneous
juxtacellular recordings). Each recording's fluorescence is converted to dF/F, averaged
onto our 20 Hz frame grid and scaled so that a single action potential has a given
amplitude relative to the noise, and real noise of our miniscope data is added: the
CNMF-E residuals (raw - denoised trace) of the example recording's cells, which carry the
1p background fluctuations and their temporal correlations (sign-flipped).

Ground-truth events: action potentials grouped into bursts (gap < burst_gap_s); a burst is
detected if an event lies within tol_s of its first spike (or within the burst). Detectors:
  frame     current pipeline: OASIS spikes above thr x noise per frame, runs of
            consecutive above-threshold frames merged into one event at the run onset
  sum<w>    OASIS spike mass summed over a sliding window of w frames (the GCaMP7f rise
            spreads one transient over 2-4 frames), peaks >= min_sep_s apart above thr x noise
  smin      OASIS with a minimum spike size s_min = thr x noise (thresholded OASIS), then
            run onsets as in 'frame'
  ar2_sum<w> AR(2) deconvolution (models the rise), then the summed detector

  conda run -n parker python benchmark_spike_detection.py
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

gt_dir = "/home/rgast/tools/Cascade/Ground_truth/DS29-GCaMP7f-m-V1"
noise_file = "/home/rgast/data/parker_data/raw_pipeline/20191205_m85_clo01_amph_rec1/" \
             "new_pipeline_outputs_p1_c0.6_pnr5_snr2_eta0.4.npz"
out_dir = "/home/rgast/data/parker_data/spike_benchmark"
fs = 20.0
burst_gap_s = 0.5               # APs closer than this form one burst (one calcium event)
tol_s = 0.25                    # detection tolerance around the burst onset
amp_snr = [1.0, 1.5, 2.0, 3.0]  # single-AP transient amplitude / noise SD
thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
min_sep_s = 0.5
seed = 1

def load_gt():
    """List of (dF/F at 20 Hz, AP times in s) per ground-truth neuron recording."""
    recs = []
    for f in sorted(glob.glob(f"{gt_dir}/*.mat")):
        c = loadmat(f, simplify_cells=True)["CAttached"]
        for x in (c if isinstance(c, list) else [c]):
            t, F = np.asarray(x["fluo_time"], float), np.asarray(x["fluo_mean"], float)
            ap = np.atleast_1d(np.asarray(x["events_AP"], float)) / 1e4   # 1e-4 s units
            if len(ap) < 5 or len(t) < 100:
                continue
            if ap.max() > t[-1] * 2:                                       # (units check)
                ap = ap / 10
            dff = F                                     # fluo_mean is dF/F (as used by CASCADE)
            ok = np.isfinite(t) & np.isfinite(dff)
            t, dff = t[ok], dff[ok]
            grid = np.arange(t[0], t[-1], 1 / fs)
            idx = np.searchsorted(t, grid)
            y = np.add.reduceat(dff, idx[:-1]) / np.diff(idx).clip(1) if len(idx) > 1 else None
            recs.append((y, ap - grid[0], os.path.basename(f)))
    return recs

def single_ap_amplitude(recs):
    """Median 20-Hz dF/F peak after isolated single APs (no other AP within 1 s)."""
    amps = []
    for y, ap, _ in recs:
        for i, s in enumerate(ap):
            if (i == 0 or s - ap[i - 1] > 1) and (i == len(ap) - 1 or ap[i + 1] - s > 1):
                k = int(s * fs)
                if 2 <= k < len(y) - 6:
                    amps.append(y[k:k + 6].max() - y[k - 2:k].mean())
    return np.median(amps)

def bursts(ap):
    """Onset and end times of AP bursts (gap < burst_gap_s)."""
    if len(ap) == 0:
        return np.zeros((0, 2))
    br = np.split(ap, np.flatnonzero(np.diff(ap) >= burst_gap_s) + 1)
    return np.array([[b[0], b[-1]] for b in br])

def deconv(y, p):
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi, GetSn
    sn = GetSn(y)
    c, bl, c1, g, sn, s, lam = constrained_foopsi(y, p=p, sn=sn, method_deconvolution="oasis")
    return s, sn, g

def deconv_smin(y, s_min):
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi, GetSn
    sn = GetSn(y)
    *_, s, lam = constrained_foopsi(y, p=1, sn=sn, s_min=s_min * sn, method_deconvolution="oasis")
    return s

def detect_frame(s, sn, thr):
    above = s > thr * sn
    on = above.copy()
    on[1:] &= ~above[:-1]
    return np.flatnonzero(on) / fs

def detect_sum(s, sn, thr, w):
    m = uniform_filter1d(s, w, origin=-(w // 2)) * w        # mass in [t, t + w)
    pk, _ = find_peaks(m, height=thr * sn, distance=max(1, int(min_sep_s * fs)))
    # event time: first frame of the window carrying spike mass
    return np.array([p + np.argmax(s[p:p + w] > 0) for p in pk]) / fs

def score(det, br):
    """Recall of bursts, precision of detections, events per detected burst."""
    hit = np.zeros(len(br), bool)
    used = np.zeros(len(det), bool)
    per = []
    for i, (a, b) in enumerate(br):
        m = (det >= a - tol_s) & (det <= b + tol_s)
        hit[i] = m.any()
        used |= m
        if m.any():
            per.append(m.sum())
    return hit.sum(), len(br), used.sum(), len(det), (np.mean(per) if per else np.nan)

def main():
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    recs = load_gt()
    a1 = single_ap_amplitude(recs)
    print(f"{len(recs)} ground-truth recordings, single-AP dF/F at 20 Hz: {a1:.3f}")
    z = np.load(noise_file)
    # real 1p noise: the sign-flipped CNMF-E residual (as the pipeline's noise null): keeps
    # its amplitude distribution and temporal correlations, but unmodelled positive
    # transients become negative and cannot cause detections
    noise = z["C"] - z["Yraw"]
    noise = noise / noise.std(1, keepdims=True)
    rows = []
    for snr in amp_snr:
        for j, (y, ap, name) in enumerate(recs):
            n0 = noise[rng.integers(len(noise))]
            st = rng.integers(0, len(n0) - len(y))
            trace = y / a1 * snr + n0[st:st + len(y)]               # noise SD = 1
            br = bursts(ap[ap < len(y) / fs])
            s1, sn1, _ = deconv(trace, 1)
            s2, sn2, _ = deconv(trace, 2)
            for thr in thresholds:
                det = {"frame": detect_frame(s1, sn1, thr)}
                for w in (3, 4, 6):
                    det[f"sum{w}"] = detect_sum(s1, sn1, thr, w)
                det["ar2_sum4"] = detect_sum(s2, sn2, thr, 4)
                det["smin"] = detect_frame(deconv_smin(trace, thr), sn1, 0.0)
                for meth, d in det.items():
                    h, nb, u, nd, per = score(d, br)
                    rows.append(dict(snr=snr, rec=j, method=meth, thr=thr, hits=h, bursts=nb,
                                     used=u, dets=nd, per_burst=per, dur_s=len(y) / fs))
        print(f"amplitude {snr:g} x noise done")
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/benchmark.csv", index=False)
    g = df.groupby(["snr", "method", "thr"]).sum(numeric_only=True)
    g["recall"] = g.hits / g.bursts
    g["precision"] = g.used / g.dets.clip(1)
    g["F1"] = 2 * g.recall * g.precision / (g.recall + g.precision)
    g["fp_per_min"] = (g.dets - g.used) / (g.dur_s / 60)
    g["per_burst"] = df.groupby(["snr", "method", "thr"]).per_burst.mean()
    g = g[["recall", "precision", "F1", "fp_per_min", "per_burst"]].round(3)
    g.to_csv(f"{out_dir}/benchmark_summary.csv")
    pd.set_option("display.width", 200)
    best = g.reset_index().sort_values("F1").groupby(["snr", "method"]).tail(1)
    print(best.sort_values(["snr", "F1"], ascending=[True, False]).to_string(index=False))
    print("\ncurrent detector (frame, thr 1.5):")
    print(g.xs(("frame", 1.5), level=("method", "thr")).to_string())


if __name__ == "__main__":
    main()
