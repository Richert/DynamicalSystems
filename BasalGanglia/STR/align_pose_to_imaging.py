"""Align the MARS pose data (20 Hz video) to the Ca2+ imaging recordings.

The video starts at the drug/vehicle (or amphetamine) injection and also covers the
10-min wait before imaging, so imaging starts ~12,000 video frames in. The camera is
TTL-triggered at 20 Hz. Per session, the offset is found by cross-correlating a speed
trace computed from the keypoints like the original pipeline's speed (body centroid,
1-s median filter, downsampled x4 to 5 Hz) with the original 5 Hz speed trace, which
is aligned to the imaging; all 4 sub-sample phases are tried, giving 20 Hz precision.

Pixels -> cm: the arena (30.48 cm diameter) spans the keypoint extent (0.01-99.99th
percentiles, as in the paper's MARS feature extraction).

Outputs (pose_alignment/): offsets per session (video frame of the first imaging
frame), peak correlation, peak distinctness, cm/px scale and the old/new speed ratio,
plus a summary figure.

  /home/rgast/conda/envs/arhmm/bin/python align_pose_to_imaging.py
"""

import os
import glob
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import median_filter

pose_root = "/home/rgast/data/parker_data/pose_data/Clozapine"      # local copy
speed_cache = "/home/rgast/data/parker_data/speed_traces_5hz.h5"    # old 5 Hz speed
out_dir = "/home/rgast/data/parker_data/pose_alignment"
drug_old = "clozapine"
fs = 20.0
arena_cm = 30.48
down = 4                          # 20 Hz -> 5 Hz
expected_offset_s = 600.0         # 10-min wait before imaging

def load_pose(f):
    """Keypoints (T, 7, 2) in pixels and node names."""
    with h5py.File(f, "r") as h:
        tr = h["tracks"][0]                              # (2, 7, T)
        nodes = [n.decode() for n in h["node_names"][...]]
    return np.transpose(tr, (2, 1, 0)), nodes

def cm_per_px(kp):
    lo = np.nanpercentile(kp.reshape(-1, 2), 0.01, axis=0)
    hi = np.nanpercentile(kp.reshape(-1, 2), 99.99, axis=0)
    return arena_cm / np.mean(hi - lo)

def speed_20hz(kp, scale):
    """Body-centroid speed (cm/s) at 20 Hz, 1-s median filtered (as the old pipeline)."""
    c = np.nanmean(kp, axis=1)
    v = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=1)] * fs * scale
    return median_filter(v, size=int(fs) + 1, mode="nearest")

def best_offset(v20, old):
    """Video frame (20 Hz) where the old 5 Hz trace starts: max Pearson correlation
    over all 5 Hz shifts and the 4 sub-sample phases."""
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    zo, n = z(old), len(old)
    best = (-np.inf, 0, None)
    curves = {}
    for p in range(down):
        v5 = v20[p:][: (len(v20) - p) // down * down].reshape(-1, down).mean(1)
        if len(v5) < n:
            continue
        # sliding correlation via cumulative sums
        cs, cs2 = np.cumsum(np.r_[0, v5]), np.cumsum(np.r_[0, v5 ** 2])
        m = (cs[n:] - cs[:-n]) / n
        sd = np.sqrt(np.maximum((cs2[n:] - cs2[:-n]) / n - m ** 2, 1e-12))
        dot = np.correlate(v5, zo, mode="valid") / n
        r = dot / sd                                   # zo has mean 0 -> mean term drops
        curves[p] = r
        k = int(np.argmax(r))
        if r[k] > best[0]:
            best = (r[k], p + down * k, (p, k))
    r_best, off, (p, k) = best
    r = curves[p]
    excl = np.abs(np.arange(len(r)) - k) > int(10 * fs / down)      # outside +/- 10 s
    second = r[excl].max() if excl.any() else np.nan
    v5 = v20[p:][: (len(v20) - p) // down * down].reshape(-1, down).mean(1)[k:k + n]
    ratio = np.median(old[old > 1]) / np.median(v5[old > 1]) if (old > 1).sum() > 20 else np.nan
    return off, r_best, second, ratio, r, v5

def main():
    os.makedirs(out_dir, exist_ok=True)
    rows, examples = [], []
    with h5py.File(speed_cache, "r") as cache:
        for f in sorted(glob.glob(f"{pose_root}/*/*/*_pose_top_v1_8.h5")):
            dose, part = f.split("/")[-3:-1]
            base = os.path.basename(f).split("_pose_top")[0]   # Amph files already end in _amph
            key = f"{drug_old}/{dose}/{base}"
            if key not in cache:
                print(f"no old speed for {key}")
                continue
            old = cache[key][...].astype(float)
            kp, _ = load_pose(f)
            scale = cm_per_px(kp)
            v20 = speed_20hz(kp, scale)
            off, r, second, ratio, curve, v5 = best_offset(v20, old)
            rows.append(dict(session=key, dose=dose, part=part, video_frames=len(kp),
                             old_frames_5hz=len(old), offset_frames=off, offset_s=off / fs,
                             r_peak=r, r_second=second, cm_per_px=scale,
                             old_over_new_speed=ratio))
            examples.append((key, curve, old, v5))
            print(f"{key:45s} offset {off:6d} frames ({off / fs:6.1f} s)  r = {r:.3f} "
                  f"(next {second:.3f})  scale {scale:.4f} cm/px  speed ratio {ratio:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/pose_imaging_offsets.csv", index=False)
    print("\n", df[["offset_s", "r_peak", "r_second", "cm_per_px", "old_over_new_speed"]]
          .describe().round(3).to_string())
    make_figure(df, examples)

def make_figure(df, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, blue, red = "#1f1f1e", "#5f5e5a", "#2a78d6", "#e34948"
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    ax = axes[0, 0]
    ax.hist(df.offset_s - expected_offset_s, bins=40, color=blue)
    ax.set(xlabel="offset - 600 s (s)", ylabel="# sessions",
           title="video frame of imaging start, relative to the expected 10 min")
    ax = axes[0, 1]
    ax.scatter(df.r_second, df.r_peak, s=18, color=blue)
    lim = [0, 1]
    ax.plot(lim, lim, color=ink2, lw=0.8)
    ax.set(xlabel="best correlation outside +/- 10 s of the peak", ylabel="peak correlation",
           title="alignment quality (points far above the diagonal = unambiguous)")
    ax = axes[1, 0]
    key, curve, old, v5 = examples[int(np.argmin([abs(r - df.r_peak.median()) for r in df.r_peak]))]
    t = np.arange(len(old)) / (fs / down)
    ax.plot(t, old, color=ink2, lw=1, label="original speed (5 Hz)")
    ax.plot(t, v5, color=red, lw=1, label="keypoint speed, aligned")
    ax.set(xlim=(0, 180), xlabel="time from imaging start (s)", ylabel="speed (cm/s)",
           title=f"example (median-quality session): {key}")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1, 1]
    ax.hist(df.old_over_new_speed.dropna(), bins=30, color=blue)
    ax.axvline(1, color=ink2, ls=":")
    ax.set(xlabel="original / keypoint speed (median, moving frames)", ylabel="# sessions",
           title="speed scale agreement (1 = same cm/s)")
    for a in axes.flat:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Pose (MARS keypoints, 20 Hz) to imaging alignment via speed cross-correlation",
                 fontsize=12, color=ink)
    fig.savefig(f"{out_dir}/pose_imaging_alignment.png", dpi=120, facecolor="white")
    plt.close(fig)
    print(f"saved {out_dir}/pose_imaging_alignment.png")


if __name__ == "__main__":
    main()
