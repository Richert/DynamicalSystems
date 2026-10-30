"""Slide figures of the raw-calcium-to-spikes pipeline (pipeline_raw_spikes.py) for one
recording (default: m85, vehicle, amphetamine recording = the pose example session).

  slide_raw_data       raw frame, CNMF-E ring-model background, raw - background, model A.C
  slide_cells          motion correction, seeding image, footprints, accepted vs rejected
                       component (event-triggered average vs footprint)
  slide_spikes         example traces with OASIS events, raster + population rate
  slide_validation     comparison with the original Parker pipeline (all recordings)

Inputs: the local copies of the recording's outputs (traces, spikes, footprints, background
model, motion-correction shifts; see pull_inputs) and short frame ranges of raw.tif, read
from the shared drive and cached locally.

  conda run -n parker python figures_raw_pipeline.py
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

import pipeline_raw_spikes as P

drive = "/mnt/kennedy_labdata/Parkerlab/results/raw_pipeline"
local = "/home/rgast/data/parker_data/raw_pipeline"
rec = os.environ.get("RECORDING", "20191205_m85_clo01_amph_rec1")
tag = "p1_c0.6_pnr5"
sel = "snr2_eta0.4"
out_dir = os.environ.get("SLIDE_DIR", "/home/rgast/OneDrive/presentations/parker/images")
um_per_px = 3.3                     # 192 x 192 movie (lens FOV ~630 um)
spike_thr = 1.5

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# style (slides: white background, Helvetica-like font as the beamer theme)
# ---------------------------------------------------------------------------

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CELL = "#f4f3ee"                    # neutral cell outline on the calcium images
ACCENT = "#eda100"                  # events / active cells
TRACE_RAW, TRACE_C = "#b5b3ac", "#2a78d6"
plt.rcParams.update({
    "font.family": "Nimbus Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "figure.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "none", "pdf.fonttype": 42})
FLUO = LinearSegmentedColormap.from_list("fluo", ["#0b0b0b", "#3b3b38", "#9c9a92", "#f4f3ee"])
DIFF = LinearSegmentedColormap.from_list("diff", ["#0b0b0b", "#0b0b0b", "#7a5200", ACCENT,
                                                  "#fff3cf"])

def save(fig, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.pdf")
    fig.savefig(f"{out_dir}/{name}.png", dpi=200)
    plt.close(fig)
    print(f"saved {out_dir}/{name}.pdf/.png")

def frame_axes(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, loc="left")

def scalebar(ax, dims, um=100, color="white"):
    n = um / um_per_px
    y, x = dims[0] - 10, dims[1] - 10 - n
    ax.plot([x, x + n], [y, y], color=color, lw=2, solid_capstyle="butt")
    ax.text(x + n / 2, y - 4, f"{um} µm", color=color, ha="center", va="bottom", fontsize=6.5)

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class Recording:
    """Outputs of one recording (local copies) and frame access (raw.tif on the drive,
    motion-corrected with the stored rigid shifts, cached locally per frame range)."""

    def __init__(self, name):
        self.name, self.dir = name, f"{local}/{name}"
        z = np.load(f"{self.dir}/new_pipeline_outputs_{tag}_{sel}.npz")
        self.C, self.Yraw, self.S, self.sn = z["C"], z["Yraw"], z["S"], z["sn"]
        self.kept, self.fs = z["component"], float(z["fs"])
        li = np.load(f"{self.dir}/lens_inputs_{tag}.npz")
        self.dims = tuple(li["dims"])
        self.A = sp.csc_matrix((li["A_data"], li["A_indices"], li["A_indptr"]),
                               shape=tuple(li["A_shape"]))
        self.b0 = li["b0"]
        ex = np.load(f"{self.dir}/cnmfe_extra_{tag}.npz")
        self.W = sp.csr_matrix((ex["W_data"], ex["W_indices"], ex["W_indptr"]),
                               shape=tuple(ex["W_shape"]))
        self.snr = ex["snr"]
        self.shifts = np.load(f"{self.dir}/mc_shifts.npy")
        self.cn = np.load(f"{self.dir}/corr_image.npy")
        self.pnr = np.load(f"{self.dir}/pnr_image.npy")
        self.eta = pd.read_csv(f"{self.dir}/eta_{tag}.csv").set_index("component")
        self.events = P.detect(self.S, self.sn, spike_thr)
        self.T = self.C.shape[1]

    def footprint(self, k):
        return self.A[:, k].toarray().reshape(self.dims, order="F")

    def frames(self, t0, t1):
        """Motion-corrected frames t0..t1 (T, H, W) float32."""
        cache = f"{self.dir}/frames_mc_{t0}_{t1}.npy"
        if os.path.isfile(cache):
            return np.load(cache)
        import tifffile
        from caiman.motion_correction import apply_shift_iteration
        raw = np.array(tifffile.memmap(f"{drive}/{self.name}/raw.tif", mode="r")[t0:t1],
                       dtype=np.float32)
        mc = np.stack([apply_shift_iteration(f, tuple(s), border_nan="copy")
                       for f, s in zip(raw, self.shifts[t0:t1])]).astype(np.float32)
        np.save(cache, mc)
        return mc

    def C_all(self, t0, t1):
        """Denoised traces of all CNMF-E components (incl. rejected) for frames t0..t1."""
        cache = f"{self.dir}/C_all_{t0}_{t1}.npy"
        if os.path.isfile(cache):
            return np.load(cache)
        import h5py
        with h5py.File(f"{drive}/{self.name}/cnmfe_{tag}.hdf5", "r") as f:
            C = f["estimates/C"][:, t0:t1]
        np.save(cache, C)
        return C

    def background(self, Y, C):
        """CNMF-E ring-model background B = b0 + W (Y - A C - b0) at ssub_B = 2 (as in
        caiman's Estimates), for frames Y (T, H, W) and all-component traces C."""
        d0, d1 = self.dims
        Yr = Y.transpose(1, 2, 0).reshape(-1, len(Y), order="F")
        R = Yr - self.A @ C - self.b0[:, None]
        R = R.reshape(d0, d1, -1, order="F")
        Rd = R.reshape(d0 // 2, 2, d1 // 2, 2, -1).mean((1, 3))
        WB = (self.W @ Rd.reshape(-1, Rd.shape[-1], order="F")).reshape(d0 // 2, d1 // 2, -1, order="F")
        B = self.b0.reshape(d0, d1, order="F")[..., None] + np.repeat(np.repeat(WB, 2, 0), 2, 1)
        return B.transpose(2, 0, 1)

    def model(self, C):
        """Neural part A C (T, H, W)."""
        return (self.A @ C).reshape(self.dims + (C.shape[1],), order="F").transpose(2, 0, 1)

def pop_rate(r, t0, t1, smooth_s=0.5):
    """Mean event rate per cell (Hz), Gaussian-smoothed."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(r.events[:, t0:t1].mean(0).astype(float) * r.fs, smooth_s * r.fs)

# ---------------------------------------------------------------------------
# slide 1: the raw data
# ---------------------------------------------------------------------------

def slide_raw_data(r, t0, t1):
    Y = r.frames(t0, t1)
    C = r.C_all(t0, t1)
    from scipy.ndimage import gaussian_filter
    # 1 s around the moment of highest population activity (single frames are dominated
    # by pixel noise: SD ~20 vs. cell transients of ~5-15 on a baseline of ~450; 1 s ~ the
    # GCaMP7f decay time)
    # moment: the 1-s window where the fluctuating raw - background matches the cell model
    # best (at the population-rate peak, a broad transient brightening -- neuropil / out-of-
    # focus light -- dominated the lower half of the field of view)
    step = slice(None, None, 10)
    static = (Y[step] - r.background(Y[step], C[:, step])).mean(0)
    best = (-1, 0)
    for c in range(10, len(Y) - 15, 10):
        ww = slice(c - 5, c + 15)
        f = gaussian_filter((Y[ww] - r.background(Y[ww], C[:, ww])).mean(0) - static, 1.0)
        m = r.model(C[:, ww]).mean(0)
        if m.max() > 0:
            best = max(best, (np.corrcoef(f.ravel(), m.ravel())[0, 1], c))
    k = best[1]
    w = slice(k - 5, k + 15)
    print(f"example moment: frame {t0 + k}, spatial r(data, model) = {best[0]:.2f}")
    # (display: raw and background smoothed alike, sigma 1 px; the ring-model background
    # has a static pixel-scale speckle, std ~10, also in caiman's own reconstruction)
    raw = gaussian_filter(Y[w].mean(0), 1.0)
    B = gaussian_filter(r.background(Y[w], C[:, w]).mean(0), 1.0)
    AC = r.model(C[:, w]).mean(0)
    # raw - background has a static offset (pixel map, std ~10) that the ring model leaves;
    # its fluctuating part (minus the mean over the window) holds the cells + noise
    sig = gaussian_filter((Y[w] - r.background(Y[w], C[:, w])).mean(0) - static, 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(6.0, 1.95), gridspec_kw=dict(wspace=0.06))
    lo, hi = np.percentile(raw, [0.5, 99.8])
    s_hi = np.percentile(AC, 99.9)
    panels = [(raw, FLUO, lo, hi, "raw frame"), (B, FLUO, lo, hi, "background"),
              (sig, DIFF, 0, s_hi, "raw − background (fluct.)"), (AC, DIFF, 0, s_hi, "cells (model)")]
    for ax, (img, cmap, a, b, title) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=a, vmax=b, interpolation="bilinear")
        frame_axes(ax, title)
    scalebar(axes[0], r.dims)
    # signal scale: what the neural part is relative to the raw fluorescence
    pct = 100 * s_hi / np.median(raw)
    for ax, txt in ((axes[0], f"0–{hi:.0f} a.u."), (axes[1], f"0–{hi:.0f} a.u."),
                    (axes[2], f"0–{s_hi:.0f} a.u. (~{pct:.0f}% of raw)"),
                    (axes[3], f"0–{s_hi:.0f} a.u.")):
        ax.text(0, -0.03, txt, transform=ax.transAxes, va="top", fontsize=6.5, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.1)
    save(fig, "pipeline_raw_data")
    return k


# ---------------------------------------------------------------------------
# slide 2: from pixels to cells
# ---------------------------------------------------------------------------

def C_row(r, k):
    """Full denoised trace of CNMF-E component k (cached)."""
    cache = f"{r.dir}/C_comp{k}.npy"
    if os.path.isfile(cache):
        return np.load(cache)
    import h5py
    with h5py.File(f"{drive}/{r.name}/cnmfe_{tag}.hdf5", "r") as f:
        c = f["estimates/C"][k]
    np.save(cache, c)
    return c

def event_triggered(r, k):
    """Event-triggered average image of component k as in the consistency filter
    (pipeline_raw_spikes.compute_eta): at the 20 largest peaks of C (>= 1 s apart), the
    frame minus the mean of the frames 0.5-1 s earlier."""
    cache = f"{r.dir}/eta_image_comp{k}.npy"
    if os.path.isfile(cache):
        return np.load(cache)
    from scipy.signal import find_peaks
    c = C_row(r, k)
    p = P.eta_params
    b0, b1 = (int(round(s * r.fs)) for s in p["base_window_s"])
    pk, _ = find_peaks(c, height=0, distance=int(p["min_sep_s"] * r.fs))
    pk = pk[pk >= b0]
    pk = pk[np.argsort(c[pk])[::-1][:p["n_events"]]]
    img = np.mean([r.frames(t - b0, t + 1)[-1] - r.frames(t - b0, t + 1)[:b0 - b1 + 1].mean(0)
                   for t in pk], 0)
    np.save(cache, img)
    for t in pk:                                   # per-event frame caches are not needed
        os.remove(f"{r.dir}/frames_mc_{t - b0}_{t + 1}.npy")
    return img

def examples(r):
    """Accepted (high SNR, typical consistency) and rejected (high SNR, passes the shape
    filter, low event-triggered consistency) example components."""
    e = r.eta.copy()
    e["snr"] = r.snr[e.index]
    acc = e[e.index.isin(r.kept) & e.eta_r.between(0.75, 0.9)].snr.idxmax()
    rej = None
    for k in e[~e.index.isin(r.kept) & (e.eta_r < 0.25)].sort_values("snr", ascending=False).index:
        area, el, sol, parts = P.footprint_shape(r.footprint(k))
        f = P.shape_filter
        if area <= f["max_area_px"] and el <= f["max_elongation"] and \
                sol >= f["min_solidity"] and parts == 1:
            rej = k
            break
    return int(acc), int(rej), e

def crop(img, y, x, h=14):
    return img[max(0, y - h):y + h + 1, max(0, x - h):x + h + 1]

def slide_cells(r):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(6.0, 2.25))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1.0, 1, 1, 0.55, 0.55], height_ratios=[1, 1],
                  wspace=0.1, hspace=0.42, left=0.075, right=0.995, top=0.84, bottom=0.17)
    # (a) motion correction (rigid shifts)
    ax = fig.add_subplot(gs[:, 0])
    t = np.arange(r.T) / r.fs / 60
    sh = r.shifts * um_per_px
    for i, (lab, col) in enumerate((("y", "#2a78d6"), ("x", "#eb6834"))):
        ax.plot(t, sh[:, i], color=col, lw=0.5, label=lab)
    lim = np.percentile(np.abs(sh), 99.9) * 1.6
    ax.set(xlabel="time (min)", ylabel="shift (µm)", xlim=(0, t[-1]), ylim=(-lim, lim))
    ax.set_xticks([0, 20, 40])
    ax.legend(frameon=False, ncol=2, loc="upper right", handlelength=1, fontsize=6.5)
    ax.set_title("motion correction", loc="left")
    # (b) local correlation image (CNMF-E seeds where correlation >= min_corr and
    # peak-to-noise >= min_pnr, in the order of their product)
    ax = fig.add_subplot(gs[:, 1])
    ax.imshow(r.cn, cmap=FLUO, vmin=np.percentile(r.cn, 1), vmax=1)
    frame_axes(ax, "correlation image")
    ax.text(0, -0.03, "pixel vs. neighbours", transform=ax.transAxes, va="top",
            fontsize=6.5, color=INK2)
    # (c) accepted footprints on the seeding image (correlation x peak-to-noise ratio)
    ax = fig.add_subplot(gs[:, 2])
    seed = r.cn * r.pnr
    ax.imshow(seed, cmap=FLUO, vmin=0, vmax=np.percentile(seed, 99.7))
    for k in r.kept:
        a = r.footprint(k)
        ax.contour(a, [0.4 * a.max()], colors=ACCENT, linewidths=0.4)
    frame_axes(ax, f"{len(r.kept)} cells")
    ax.text(0, -0.03, "on the seeding image (corr. × PNR)", transform=ax.transAxes,
            va="top", fontsize=6.5, color=INK2)
    # (d) accepted vs rejected: footprint and event-triggered average
    acc, rej, e = examples(r)
    for row, (k, lab) in enumerate(((acc, "accepted"), (rej, "rejected"))):
        a = r.footprint(k)
        y, x = np.unravel_index(np.argmax(a), r.dims)
        img = event_triggered(r, k)
        for col, (im, cmap, ttl) in enumerate(((a, FLUO, "footprint"),
                                               (img, DIFF, "at its events"))):
            ax = fig.add_subplot(gs[row, 3 + col])
            c = crop(im, y, x)
            ax.imshow(c, cmap=cmap, vmin=0 if col else None,
                      vmax=np.percentile(c, 99.5) if col else None, interpolation="bilinear")
            if col:
                ax.contour(crop(a, y, x), [0.4 * a.max()], colors=CELL, linewidths=0.6)
            frame_axes(ax, ttl if row == 0 else None)
            if col == 1:
                ax.text(1.0, -0.06, f"{lab}  (r = {e.eta_r[k]:.2f})", transform=ax.transAxes,
                        ha="right", va="top", fontsize=6.5, color=INK2)
    fig.text(0.763, 0.93, "consistency test", fontsize=8.5, color=INK, ha="left")
    for x, l in ((0.005, "a"), (0.265, "b"), (0.495, "c"), (0.735, "d")):
        fig.text(x, 0.93, l, fontsize=10, fontweight="bold", color=INK)
    save(fig, "pipeline_cells")
    print(f"examples: accepted {acc}, rejected {rej}")


# ---------------------------------------------------------------------------
# slide 3: from fluorescence to spikes
# ---------------------------------------------------------------------------

pose_dir = "/home/rgast/data/parker_data/pose_arhmm_v5"
pose_session = "20191205_m085_clo01_amph"      # pose features of the same session

def speed_trace(t0, t1):
    f = pd.read_csv(f"{pose_dir}/features_{pose_session}.csv.gz", usecols=["speed"])
    return f.speed.values[t0:t1]

def slide_spikes(r, t0, t1, n_ex=5):
    from matplotlib.gridspec import GridSpec
    ev = r.events[:, t0:t1]
    # example cells: a typical number of events in the window (bright cells' transients are
    # often split into many events, see the summary), highest SNR among those
    n = ev.sum(1)
    snr = r.snr[r.kept]
    cand = np.flatnonzero((n >= 3) & (n <= 10))
    cand = cand[np.argsort(snr[cand])[::-1]][:4 * n_ex]
    ex = sorted(cand[np.linspace(0, len(cand) - 1, n_ex).round().astype(int)],
                key=lambda i: -n[i])
    fig = plt.figure(figsize=(6.0, 2.6))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1.15], height_ratios=[2.2, 0.55, 0.55],
                  wspace=0.18, hspace=0.12, left=0.035, right=0.99, top=0.86, bottom=0.14)
    t = np.arange(t1 - t0) / r.fs
    # (a) example traces: raw extracted trace, denoised trace, OASIS events
    ax = fig.add_subplot(gs[:, 0])
    for j, i in enumerate(ex):
        y0 = -j * 1.25
        raw, c = r.Yraw[i, t0:t1], r.C[i, t0:t1]
        sc = np.percentile(raw, 99.9) - np.percentile(raw, 5)
        ax.plot(t, y0 + (raw - np.percentile(raw, 5)) / sc, color=TRACE_RAW, lw=0.5)
        ax.plot(t, y0 + (c - np.percentile(raw, 5)) / sc, color=TRACE_C, lw=0.8)
        ts = t[ev[i]]
        ax.plot(ts, np.full(len(ts), y0 - 0.12), "|", color=ACCENT, ms=4, mew=1.2)
    ax.set_xlim(0, t[-1])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time (s)")
    ax.legend(handles=[plt.Line2D([], [], color=TRACE_RAW, lw=1, label="raw trace"),
                       plt.Line2D([], [], color=TRACE_C, lw=1, label="denoised"),
                       plt.Line2D([], [], color=ACCENT, marker="|", ls="", ms=5, mew=1.2,
                                  label="inferred spikes")],
              frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.04),
              handlelength=1.2, columnspacing=1, borderaxespad=0)
    # (b) raster of all cells (sorted by event count), population rate, speed
    ax = fig.add_subplot(gs[0, 1])
    order = np.argsort(n)[::-1]
    yy, xx = np.nonzero(ev[order])
    ax.scatter(t[xx], yy, s=0.6, color=INK, lw=0, rasterized=True)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(len(order), -1)
    ax.set_ylabel("cell", labelpad=1)
    ax.tick_params(labelbottom=False)
    ax.set_title(f"all {len(order)} cells", loc="left")
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax)
    ax2.plot(t, pop_rate(r, t0, t1), color=ACCENT, lw=0.9)
    ax2.set_ylabel("rate\n(Hz)", labelpad=1)
    ax2.tick_params(labelbottom=False)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax)
    ax3.plot(t, speed_trace(t0, t1), color=INK2, lw=0.7)
    ax3.set_ylabel("speed\n(cm/s)", labelpad=1)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax3.set_xlabel("time (s)")
    for x, l in ((0.005, "a"), (0.47, "b")):
        fig.text(x, 0.92, l, fontsize=10, fontweight="bold", color=INK)
    save(fig, "pipeline_spikes")


if __name__ == "__main__":
    r = Recording(rec)
    # the 120-s example window of the pose figures (pose frame = imaging frame, 20 Hz)
    t0, t1 = 6800, 9200
    slide_raw_data(r, t0, t1)
    slide_cells(r)
    slide_spikes(r, t0, t1)
