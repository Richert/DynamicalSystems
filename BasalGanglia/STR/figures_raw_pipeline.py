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
tag = "p1_c0.6_pnr5_nrg0.999"      # final CNMF-E fit (full run)
sel = "snr2_sz_rim5_hyb0.3-0.2_ves3_off6_fedge1.05"   # final component selection
sel_base = "snr2_eta0.4"            # (fallback: derived from these outputs, see Recording)
out_dir = os.environ.get("SLIDE_DIR", "/home/rgast/OneDrive/presentations/parker/images")
um_per_px = 3.3                     # 192 x 192 movie (lens FOV ~630 um)
# spike detection as in the final full run (summed over 3 frames, 1.5 x noise)
P.detect_method, P.sum_frames, spike_thr = "sum", 3, 1.5

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

def save_steps(fig, steps, name):
    """Animation frames: steps = list of sets of artists visible at that step; every artist in
    any step is hidden in the other steps, all other artists are always shown.
    Writes <name>_s<k>.pdf/.png (k = 1 ...)."""
    managed = set().union(*steps)
    for k, vis in enumerate(steps, 1):
        for art in managed:
            art.set_visible(art in vis)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(f"{out_dir}/{name}_s{k}.pdf")
        fig.savefig(f"{out_dir}/{name}_s{k}.png", dpi=150)
    for art in managed:
        art.set_visible(True)
    print(f"saved {out_dir}/{name}_s1..{len(steps)}")

def outline_paths(r, comps, frac=0.4):
    """(x, y) outline polygons of components (largest contour at frac x max)."""
    from skimage.measure import find_contours
    out = []
    for k in comps:
        a = r.footprint(k)
        c = find_contours(a, frac * a.max()) if a.max() > 0 else []
        out.append(max(c, key=len)[:, ::-1] if len(c) else np.zeros((0, 2)))
    return out

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
        li = np.load(f"{self.dir}/lens_inputs_{tag}.npz")
        self.dims = tuple(li["dims"])
        self.A = sp.csc_matrix((li["A_data"], li["A_indices"], li["A_indptr"]),
                               shape=tuple(li["A_shape"]))
        self.b0 = li["b0"]
        ex = np.load(f"{self.dir}/cnmfe_extra_{tag}.npz")
        self.W = sp.csr_matrix((ex["W_data"], ex["W_indices"], ex["W_indptr"]),
                               shape=tuple(ex["W_shape"]))
        self.snr = ex["snr"]
        self.eta = pd.read_csv(f"{self.dir}/eta_{tag}.csv").set_index("component")
        f = f"{self.dir}/new_pipeline_outputs_{tag}_{sel}.npz"
        if os.path.isfile(f):
            z = np.load(f)
            keep = np.arange(len(z["component"]))
        else:
            # not yet rerun: the selection only removes cells and each cell is deconvolved
            # on its own, so the new outputs are the base outputs restricted to the cells
            # the pipeline's selection keeps (same functions, mock CNMF-E estimates)
            import types
            z = np.load(f"{self.dir}/new_pipeline_outputs_{tag}_{sel_base}.npz")
            P.selection_cli(["x", "--min_snr", "2", "--min_eta", "0.4", "--size_rel",
                             "--lens_rim", "5"])
            P.eta = self.eta
            est = types.SimpleNamespace(A=self.A, SNR_comp=self.snr, b0=self.b0,
                                        r_values=np.zeros(self.A.shape[1]),
                                        idx_components=np.flatnonzero(self.snr >= 3))
            idx, _ = P.accepted_components(types.SimpleNamespace(estimates=est), self.dims)
            keep = np.flatnonzero(np.isin(z["component"], idx))
        self.C, self.Yraw, self.S = z["C"][keep], z["Yraw"][keep], z["S"][keep]
        self.sn, self.kept, self.fs = z["sn"][keep], z["component"][keep], float(z["fs"])
        self.shifts = np.load(f"{self.dir}/mc_shifts.npy")
        self.cn = np.load(f"{self.dir}/corr_image.npy")
        self.pnr = np.load(f"{self.dir}/pnr_image.npy")
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
        cache = f"{self.dir}/C_all_{tag}_{t0}_{t1}.npy"
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
    cache = f"{r.dir}/C_comp{k}_{tag}.npy"
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
    cache = f"{r.dir}/eta_image_comp{k}_{tag}.npy"
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
    ax.text(0, -0.03, "on the seeding image", transform=ax.transAxes,
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

dataset = os.environ.get("DATASET", "/home/rgast/data/parker_data/dataset/clozapine_spn.h5")

def speed_trace(t0, t1):
    """Torso speed per imaging frame (data file: pose aligned with the fixed video offset)."""
    import h5py
    with h5py.File(dataset, "r") as h:
        return h[f"{rec[:-5]}/{rec[-4:]}/behaviour/speed"][t0:t1]

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


# ---------------------------------------------------------------------------
# slide 4: validation against the original Parker pipeline (all recordings)
# ---------------------------------------------------------------------------

def slide_validation():
    """Per recording (evaluation summaries of the final selection): cells found by both
    pipelines, share of the old pipeline's active neurons recovered and the trace
    correlation of the matched pairs, recall of the old events (within 0.25 s)."""
    rows = []
    for f in glob.glob(f"{local}/*_rec?/summary_{tag}_thr1.5_sum3_{sel}.json"):
        d = json.load(open(f))
        name = os.path.basename(os.path.dirname(f))
        rows.append(dict(rec=name, long=name.endswith("rec1"), n_old=d["n_old_total"],
                         n_old_active=d["n_old_active"], n_new=d["n_new"],
                         matched=d["frac_active_old_matched"], r=d["r_matched_median"],
                         recall=d["recall_mean"], rate_new=d["rate_new_median_hz"],
                         rate_old=d["rate_old_median_hz"]))
    v = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.2), gridspec_kw=dict(wspace=0.55))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.2)
    col = {False: "#2a78d6", True: "#eb6834"}
    # (a) cells per recording
    ax = axes[0]
    for lg in (False, True):
        w = v[v.long == lg]
        ax.scatter(w.n_old, w.n_new, s=10, color=col[lg], lw=0, alpha=0.8,
                   label="15 min" if not lg else "45 min")
    m = max(v.n_old.max(), v.n_new.max()) * 1.05
    ax.plot([0, m], [0, m], color=AXIS, lw=0.8, ls=(0, (3, 2)))
    ax.set(xlim=(0, m), ylim=(0, m), xlabel="cells, original pipeline", ylabel="cells, new pipeline")
    ax.legend(frameon=False, handletextpad=0.2, loc="upper left", fontsize=6.5)
    ax.set_title(f"cell yield ({np.median(v.n_new / v.n_old):.1f}× median)", loc="left")
    # (b) recovery of the old active neurons and trace agreement
    def strip(ax, cols, labels, title, ylabel, ylim):
        rng = np.random.default_rng(1)
        for i, c in enumerate(cols):
            y = v[c].dropna().values
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=6, color=INK2, alpha=0.5, lw=0)
            ax.plot([i - 0.28, i + 0.28], [np.median(y)] * 2, color=ACCENT, lw=2)
            ax.text(i + 0.32, np.median(y), f"{np.median(y):.2f}", va="center", fontsize=6.5,
                    color=INK)
        ax.set_xticks(range(len(cols)), labels)
        ax.set(ylim=ylim, ylabel=ylabel, xlim=(-0.5, len(cols) - 0.2))
        ax.set_title(title, loc="left")
        ax.tick_params(axis="x", length=0)
    strip(axes[1], ["matched", "r"], ["recovered", "trace r"],
          "original active neurons", "share / correlation", (0, 1.02))
    strip(axes[2], ["recall"], ["recall"], "original events", "share found (±0.25 s)", (0, 1.02))
    ax = axes[2]
    ax.text(0.98, 0.05, f"{len(v)} recordings\nnoise events < 1%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color=INK2)
    for x, l in ((0.005, "a"), (0.355, "b"), (0.69, "c")):
        fig.text(x, 0.92, l, fontsize=10, fontweight="bold", color=INK)
    save(fig, "pipeline_validation")
    print(v[["n_old", "n_new", "matched", "r", "recall"]].median().round(3).to_dict())


# ---------------------------------------------------------------------------
# slide: the standard CaImAn pipeline (flow chart with one image per stage)
# ---------------------------------------------------------------------------

def slide_caiman_overview(r, t0, t1):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    Y = r.frames(t0, t0 + 200)
    fig = plt.figure(figsize=(6.15, 2.75))
    n = 5
    w, gap, x0 = 0.168, 0.031, 0.012
    stages = [
        ("raw movie", "1p miniscope, 20 Hz"),
        ("motion correction", "NoRMCorre (rigid)"),
        ("cell detection", "CNMF-E: seeds from\ncorrelation × PNR"),
        ("demixing", "footprints + traces,\nring-model background"),
        ("spike inference", "OASIS deconvolution\n(AR(1))"),
    ]
    groups = []
    for i, (title, sub) in enumerate(stages):
        x = x0 + i * (w + gap)
        box = fig.add_axes([x, 0.06, w, 0.88])
        groups.append({box})
        box.axis("off")
        box.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06",
                                     transform=box.transAxes, fc="#f4f3ee", ec=AXIS, lw=0.8))
        box.text(0.5, 0.95, title, ha="center", va="top", fontsize=8, fontweight="bold", color=INK,
                 transform=box.transAxes)
        box.text(0.5, 0.05, sub, ha="center", va="bottom", fontsize=6.3, color=INK2,
                 transform=box.transAxes)
        ax = fig.add_axes([x + 0.012, 0.3, w - 0.024, 0.5])
        groups[-1].add(ax)
        frame_axes(ax)
        if i == 0:
            ax.imshow(Y[0], cmap=FLUO, vmin=np.percentile(Y[0], 1), vmax=np.percentile(Y[0], 99.8))
        elif i == 1:
            t = np.arange(len(r.shifts)) / r.fs / 60
            ax.plot(t, r.shifts[:, 0] * um_per_px, color="#2a78d6", lw=0.4)
            ax.plot(t, r.shifts[:, 1] * um_per_px, color="#eb6834", lw=0.4)
            ax.set_xlim(0, t[-1]); ax.set_ylim(-6, 6)
            ax.text(0.03, 0.95, "shift (µm), 45 min", transform=ax.transAxes, va="top", fontsize=5.5, color=INK2)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(AXIS)
        elif i == 2:
            seed = r.cn * r.pnr
            ax.imshow(seed, cmap=FLUO, vmin=0, vmax=np.percentile(seed, 99.7))
        elif i == 3:
            ax.imshow(r.cn, cmap=FLUO, vmin=np.percentile(r.cn, 1), vmax=1)
            for k in r.kept:
                a = r.footprint(k)
                ax.contour(a, [0.4 * a.max()], colors=ACCENT, linewidths=0.3)
        else:
            # cell with a clear transient: largest denoised peak relative to its noise, 3-8 events
            nev = r.events[:, t0:t1].sum(1)
            pk_snr = r.C[:, t0:t1].max(1) / r.sn
            i_ex = int(np.argmax(np.where((nev >= 3) & (nev <= 8), pk_snr, -1)))
            tp = t0 + int(np.argmax(r.C[i_ex, t0:t1]))
            a, b_ = max(t0, tp - int(6 * r.fs)), min(t1, tp + int(14 * r.fs))
            tt = np.arange(b_ - a) / r.fs
            y = r.Yraw[i_ex, a:b_]; c = r.C[i_ex, a:b_]
            sc = np.percentile(y, 99.9) - np.percentile(y, 5)
            ax.plot(tt, (y - np.percentile(y, 5)) / sc, color=TRACE_RAW, lw=0.4)
            ax.plot(tt, (c - np.percentile(y, 5)) / sc, color=TRACE_C, lw=0.8)
            ts = tt[r.events[i_ex, a:b_]]
            ax.plot(ts, np.full(len(ts), -0.12), "|", color=ACCENT, ms=5, mew=1.2)
            ax.set_xlim(0, tt[-1]); ax.set_ylim(-0.25, 1.15)
            ax.text(0.03, 0.95, "trace + spikes, 20 s", transform=ax.transAxes, va="top", fontsize=5.5, color=INK2)
        if i < n - 1:
            arr = FancyArrowPatch((x + w + 0.003, 0.55), (x + w + gap - 0.003, 0.55),
                                  transform=fig.transFigure, arrowstyle="-|>",
                                  mutation_scale=9, color=INK2, lw=1)
            fig.patches.append(arr)
            groups[-1].add(arr)
    save(fig, "pipeline_caiman_overview")
    # build-up: box i appears with the arrow leading into it
    steps, vis = [], set()
    for i, g in enumerate(groups):
        boxes = {a_ for a_ in g if not isinstance(a_, FancyArrowPatch)}
        into = {a_ for a_ in (groups[i - 1] if i else set()) if isinstance(a_, FancyArrowPatch)}
        vis = vis | boxes | into
        steps.append(set(vis))
    save_steps(fig, steps, "pipeline_caiman_overview")


# ---------------------------------------------------------------------------
# slide: post-processing on top of CaImAn (component selection)
# ---------------------------------------------------------------------------

RULES = [  # (key, label, criterion, colour)
    ("snr", "low SNR", "transient SNR < 2", "#8c8a84"),
    ("shape", "odd shape", "too large, elongated\nor split", "#e87ba4"),
    ("size", "oversized", "> 3x the median\ncell area", "#7a4fa3"),
    ("consistency", "inconsistent", "its events don't light\nup its footprint", "#e34948"),
    ("vessel", "blood vessel", "centred on a vessel\n(baseline image)", "#2a78d6"),
    ("offpeak", "off-peak", "> 6 px from any\ncorrelation peak", "#eda100"),
    ("edge", "edge piece", "on the edge of a\nneighbouring cell", "#1fa187"),
]

def selection_stages(r):
    """CNMF-E component -> first selection step that removes it (or 'kept'), by replaying the
    pipeline's selection with one rule more at a time (same functions, local inputs)."""
    import types
    ex = np.load(f"{r.dir}/cnmfe_extra_{tag}.npz")
    est = types.SimpleNamespace(A=r.A, SNR_comp=r.snr, b0=r.b0, r_values=ex["r_values"],
                                idx_components=ex["idx_components"])
    cnm = types.SimpleNamespace(estimates=est)
    P.eta, P.pnr_img = r.eta, r.pnr
    stage = {}
    for k in range(r.A.shape[1]):
        if r.snr[k] < 2:
            stage[k] = "snr"
    flags = [("shape", []), ("size", ["--size_rel"]), ("rim", ["--lens_rim", "5"]),
             ("consistency", ["--hybrid", "0.3", "0.2"]), ("vessel", ["--vessel", "3"]),
             ("offpeak", ["--offpeak", "6"]), ("edge", ["--frag_edge", "1.05"])]
    argv, prev = ["x", "--min_snr", "2"], set(k for k in range(r.A.shape[1]) if k not in stage)
    for key, f in flags:
        argv = argv + f
        P.selection_cli(list(argv))
        idx, _ = P.accepted_components(cnm, r.dims, r.cn)
        for k in prev - set(idx):
            stage[k] = key
        prev = set(idx)
    for k in prev:
        stage[k] = "kept"
    assert prev == set(r.kept), "selection replay differs from the stored selection"
    return stage

def slide_postprocessing(r, pick=None):
    """pick: rule -> component shown as its example (default: highest SNR away from the edge;
    the defaults for the example recording were chosen by eye from all candidates)."""
    if pick is None and r.name == "20191205_m85_clo01_amph_rec1" and tag == "p1_c0.6_pnr5_nrg0.999":
        pick = {"size": 1047, "offpeak": 523}
    pick = pick or {}
    from matplotlib.gridspec import GridSpec
    stage = selection_stages(r)
    col = {k: c for k, _, _, c in RULES}
    seed = r.cn * r.pnr
    fig = plt.figure(figsize=(6.0, 3.05))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1.55, 0.9, 0.62, 0.62, 0.62],
                  height_ratios=[1, 1], wspace=0.12, hspace=0.62, left=0.005, right=0.995,
                  top=0.88, bottom=0.12)
    # (a) all CNMF-E components on the seeding image, coloured by the step removing them
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(seed, cmap=FLUO, vmin=0, vmax=np.percentile(seed, 99.7))
    for k, st in stage.items():
        if st == "snr":
            continue                                       # (most: noise / background)
        a = r.footprint(k)
        ax.contour(a, [0.4 * a.max()], colors=[CELL if st == "kept" else col[st]],
                   linewidths=0.35 if st == "kept" else 0.7)
    frame_axes(ax, None)
    ax.set_title("CNMF-E components (SNR ≥ 2)", loc="left")
    scalebar(ax, r.dims)
    # (b) counts per step
    ax = fig.add_subplot(gs[:, 1])
    n_all = len(stage)
    keys = [k for k, *_ in RULES]
    counts = [sum(v == k for v in stage.values()) for k in keys]
    y = np.arange(len(keys) + 1)
    n_kept = sum(v == "kept" for v in stage.values())
    ax.barh(y[:-1], counts, color=[col[k] for k in keys], height=0.68)
    ax.barh(y[-1], n_kept, color="#c9c7bf", height=0.68)
    for yi, c, l in zip(y, counts + [n_kept], [l for _, l, _, _ in RULES] + ["kept cells"]):
        inside = c > 250
        ax.text(c - 12 if inside else c + 12, yi, f"{c}  {l}", va="center", fontsize=6.8,
                color="white" if l == "low SNR" else INK, ha="right" if inside else "left")
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts + [n_kept]) * 1.05)
    ax.set_xticks([0, 250, 500])
    ax.set_xlabel("components")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{n_all} → {n_kept}", loc="left")
    # (c) one example per added rule (highest SNR), on the correlation image (vessel: baseline)
    ex_rules = [R_ for R_ in RULES if R_[0] in ("shape", "size", "consistency", "vessel",
                                                 "offpeak", "edge")]
    b0 = r.b0.reshape(r.dims, order="F")
    for i, (key, lab, crit, c) in enumerate(ex_rules):
        ax = fig.add_subplot(gs[i // 3, 2 + i % 3])
        ks = [k for k, v in stage.items() if v == key]
        if not ks:
            frame_axes(ax, lab)
            continue
        h = 16 if key in ("size", "shape", "vessel") else 11
        inner = [k for k in ks if min(*np.unravel_index(np.argmax(r.footprint(k)), r.dims),
                                      *(np.array(r.dims) - 1 - np.unravel_index(
                                          np.argmax(r.footprint(k)), r.dims))) >= h]
        k = pick.get(key) or max(inner or ks, key=lambda k: r.snr[k])   # (highest SNR, not at the edge)
        a = r.footprint(k)
        yy, xx = np.unravel_index(np.argmax(a), r.dims)
        yy, xx = int(np.clip(yy, h, r.dims[0] - 1 - h)), int(np.clip(xx, h, r.dims[1] - 1 - h))
        img = b0 if key == "vessel" else r.cn
        ci = crop(img, yy, xx, h)
        ax.imshow(ci, cmap="gray" if key == "vessel" else FLUO, interpolation="bilinear",
                  vmin=np.percentile(ci, 1), vmax=np.percentile(ci, 99.5))
        for kk in r.kept:                                  # kept neighbours
            b = r.footprint(kk)
            if abs(np.unravel_index(np.argmax(b), r.dims)[0] - yy) <= h + 6 and \
                    abs(np.unravel_index(np.argmax(b), r.dims)[1] - xx) <= h + 6:
                ax.contour(crop(b, yy, xx, h), [0.4 * b.max()], colors=CELL, linewidths=0.5)
        ax.contour(crop(a, yy, xx, h), [0.4 * a.max()], colors=c, linewidths=1.1)
        frame_axes(ax, None)
        ax.set_title(lab, loc="left", fontsize=7.5, color=c, fontweight="bold", pad=2)
        ax.text(0, -0.04, crit, transform=ax.transAxes, va="top", fontsize=6.2, color=INK2,
                linespacing=1.1)
    for x, l in ((0.005, "a"), (0.345, "b"), (0.565, "c")):
        fig.text(x, 0.955, l, fontsize=10, fontweight="bold", color=INK)
    save(fig, "pipeline_postprocessing")
    print({k: sum(v == k for v in stage.values()) for k in keys + ["rim", "kept"]})


# ---------------------------------------------------------------------------
# animated slides (build-up frames)
# ---------------------------------------------------------------------------

def slide_spikes_steps(r, t0, t1, n_ex=5):
    """From fluorescence to spikes, built up: raw traces -> denoised -> inferred spikes ->
    raster of all cells -> population rate (Gaussian-smoothed events)."""
    from matplotlib.gridspec import GridSpec
    ev = r.events[:, t0:t1]
    n = ev.sum(1)
    snr = r.snr[r.kept]
    cand = np.flatnonzero((n >= 3) & (n <= 10))
    cand = cand[np.argsort(snr[cand])[::-1]][:4 * n_ex]
    ex = sorted(cand[np.linspace(0, len(cand) - 1, n_ex).round().astype(int)], key=lambda i: -n[i])
    fig = plt.figure(figsize=(6.15, 2.75))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.15], height_ratios=[2.2, 0.75],
                  wspace=0.2, hspace=0.12, left=0.035, right=0.99, top=0.84, bottom=0.15)
    t = np.arange(t1 - t0) / r.fs
    ax = fig.add_subplot(gs[:, 0])
    raw_l, den_l, spk_l = set(), set(), set()
    for j, i in enumerate(ex):
        y0 = -j * 1.25
        raw, c = r.Yraw[i, t0:t1], r.C[i, t0:t1]
        sc = np.percentile(raw, 99.9) - np.percentile(raw, 5)
        raw_l |= set(ax.plot(t, y0 + (raw - np.percentile(raw, 5)) / sc, color=TRACE_RAW, lw=0.5))
        den_l |= set(ax.plot(t, y0 + (c - np.percentile(raw, 5)) / sc, color=TRACE_C, lw=0.8))
        ts = t[ev[i]]
        spk_l |= set(ax.plot(ts, np.full(len(ts), y0 - 0.12), "|", color=ACCENT, ms=4, mew=1.2))
    ax.set_xlim(0, t[-1])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time (s)")
    leg = []
    for x, (kind, lab) in zip((0.035, 0.17, 0.29), (("raw", "raw trace"), ("den", "denoised"),
                                                  ("spk", "inferred spikes"))):
        if kind == "spk":
            h = fig.add_artist(plt.Line2D([x, x], [0.9, 0.95], color=ACCENT, lw=1.6, transform=fig.transFigure))
        else:
            h = fig.add_artist(plt.Line2D([x, x + 0.025], [0.925] * 2, color=TRACE_RAW if kind == "raw" else TRACE_C,
                                          lw=1.2, transform=fig.transFigure))
        tx = fig.text(x + (0.012 if kind == "spk" else 0.032), 0.925, lab, va="center", fontsize=7, color=INK2)
        leg.append({h, tx})
    axr = fig.add_subplot(gs[0, 1])
    order = np.argsort(n)[::-1]
    yy, xx = np.nonzero(ev[order])
    axr.scatter(t[xx], yy, s=0.6, color=INK, lw=0, rasterized=True)
    axr.set_xlim(0, t[-1])
    axr.set_ylim(len(order), -1)
    axr.set_ylabel("cell", labelpad=1)
    axr.tick_params(labelbottom=False)
    axr.set_title(f"spikes of all {len(order)} cells", loc="left")
    axp = fig.add_subplot(gs[1, 1], sharex=axr)
    axp.plot(t, pop_rate(r, t0, t1), color=ACCENT, lw=0.9)
    axp.set_ylabel("rate (Hz)", labelpad=2, fontsize=7)
    axp.yaxis.set_major_locator(plt.MaxNLocator(2))
    axp.set_xlabel("time (s)")
    la = fig.text(0.005, 0.92, "a", fontsize=10, fontweight="bold", color=INK)
    lb = fig.text(0.47, 0.92, "b", fontsize=10, fontweight="bold", color=INK)
    base = {ax, la}
    s1 = base | raw_l | leg[0]
    s2 = s1 | den_l | leg[1]
    s3 = s2 | spk_l | leg[2]
    s4 = s3 | {axr, lb}
    s5 = s4 | {axp}
    save_steps(fig, [s1, s2, s3, s4, s5], "pipeline_spikes_anim")
    plt.close(fig)

def slide_postprocessing_steps(r, pick=None):
    """Post-processing on top of CaImAn, built up rule by rule: all CNMF-E components (grey)
    -> each rule colours the components it removes, adds its bar (b) and its example (c) ->
    final: dropped components blue, kept cells dark yellow."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.collections import LineCollection
    stage = selection_stages(r)
    if pick is None and r.name == "20191205_m85_clo01_amph_rec1" and tag == "p1_c0.6_pnr5_nrg0.999":
        pick = {"size": 1047, "offpeak": 523}
    pick = pick or {}
    col = {k: c for k, _, _, c in RULES}
    col["snr"] = "#9fb7d6"
    lab = {k: l for k, l, _, _ in RULES}
    crit = {k: c for k, _, c, _ in RULES}
    crit["snr"] = "transient SNR < 2"
    keys = [k for k, *_ in RULES]
    comps = sorted(stage)
    paths = dict(zip(comps, outline_paths(r, comps)))
    DROP, KEEP, BASE = "#3f7fc6", "#d9a400", "#cfcdc4"
    seed = r.cn * r.pnr
    fig = plt.figure(figsize=(6.15, 2.85))
    gs = GridSpec(2, 6, figure=fig, width_ratios=[1.55, 0.85, 0.5, 0.5, 0.5, 0.5],
                  height_ratios=[1, 1], wspace=0.12, hspace=0.62, left=0.005, right=0.995,
                  top=0.87, bottom=0.11)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(seed, cmap=FLUO, vmin=0, vmax=np.percentile(seed, 99.7))
    frame_axes(ax, None)
    scalebar(ax, r.dims)
    lc = lambda ks, c, lw: ax.add_collection(LineCollection([paths[k] for k in ks if len(paths[k])],
                                                            colors=c, linewidths=lw))
    base = lc(comps, BASE, 0.35)
    rule_lc = {k: lc([c for c, v in stage.items() if v == k], col[k], 0.8) for k in keys}
    dropped = [c for c, v in stage.items() if v != "kept"]
    kept = [c for c, v in stage.items() if v == "kept"]
    fin_drop, fin_keep = lc(dropped, DROP, 0.55), lc(kept, KEEP, 0.8)
    t_all = ax.set_title(f"all {len(comps)} CNMF-E components", loc="left")
    t_fin = ax.text(0, 1.02, f"{len(kept)} kept (yellow), {len(dropped)} dropped (blue)",
                    transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom")
    t_all_txt = ax.title
    # (b) bars
    axb = fig.add_subplot(gs[:, 1])
    cnt = {k: sum(v == k for v in stage.values()) for k in keys}
    n_kept = len(kept)
    order = keys + ["kept"]
    y = {k: i for i, k in enumerate(order)}
    bars = {}
    for k in order:
        c = cnt[k] if k != "kept" else n_kept
        bc = axb.barh(y[k], c, color=col.get(k, KEEP), height=0.68)
        inside = c > 250
        tx = axb.text(c - 12 if inside else c + 12, y[k], f"{c}  {lab.get(k, 'kept cells')}", va="center",
                      fontsize=6.6, color="white" if inside and k != "kept" else INK,
                      ha="right" if inside else "left")
        bars[k] = set(bc.patches) | {tx}
    axb.set_yticks([])
    axb.set_ylim(len(order) - 0.4, -0.6)
    axb.set_xlim(0, max(list(cnt.values()) + [n_kept]) * 1.05)
    axb.set_xticks([0, 250, 500])
    axb.set_xlabel("components")
    for sp in ("top", "right", "left"):
        axb.spines[sp].set_visible(False)
    # (c) examples: one per rule + a kept cell
    acc, _, _ = examples(r)
    b0 = r.b0.reshape(r.dims, order="F")
    exa = {}
    for i, k in enumerate(order):
        ax_ = fig.add_subplot(gs[i // 4, 2 + i % 4])
        ks = [c for c, v in stage.items() if v == k] if k != "kept" else [acc]
        h = 16 if k in ("size", "shape", "vessel") else 11
        inner = [c for c in ks if min(*np.unravel_index(np.argmax(r.footprint(c)), r.dims),
                                      *(np.array(r.dims) - 1 - np.unravel_index(np.argmax(r.footprint(c)), r.dims))) >= h]
        if k == "snr":
            pool = inner or ks
            c0 = sorted(pool, key=lambda c: r.snr[c])[len(pool) // 2]
        else:
            c0 = pick.get(k) or max(inner or ks, key=lambda c: r.snr[c])
        a = r.footprint(c0)
        yy, xx = np.unravel_index(np.argmax(a), r.dims)
        yy, xx = int(np.clip(yy, h, r.dims[0] - 1 - h)), int(np.clip(xx, h, r.dims[1] - 1 - h))
        img = b0 if k == "vessel" else r.cn
        ci = crop(img, yy, xx, h)
        ax_.imshow(ci, cmap="gray" if k == "vessel" else FLUO, interpolation="bilinear",
                   vmin=np.percentile(ci, 1), vmax=np.percentile(ci, 99.5))
        for kk in kept:
            if kk == c0:
                continue
            bb = r.footprint(kk)
            p_ = np.unravel_index(np.argmax(bb), r.dims)
            if abs(p_[0] - yy) <= h + 6 and abs(p_[1] - xx) <= h + 6:
                ax_.contour(crop(bb, yy, xx, h), [0.4 * bb.max()], colors=CELL, linewidths=0.4)
        ax_.contour(crop(a, yy, xx, h), [0.4 * a.max()], colors=col.get(k, KEEP), linewidths=1.1)
        frame_axes(ax_, None)
        ax_.set_title(lab.get(k, "kept cell"), loc="left", fontsize=7, color=col.get(k, KEEP),
                      fontweight="bold", pad=2)
        ax_.text(0, -0.04, crit.get(k, "passes every rule"), transform=ax_.transAxes, va="top",
                 fontsize=5.6, color=INK2, linespacing=1.1)
        exa[k] = {ax_}
    for x, l in ((0.005, "a"), (0.33, "b"), (0.5, "c")):
        fig.text(x, 0.955, l, fontsize=10, fontweight="bold", color=INK)
    steps, vis = [], {base, t_all_txt}
    steps.append(set(vis))
    for k in keys:
        vis = vis | {rule_lc[k]} | bars[k] | exa[k]
        steps.append(set(vis))
    final = (vis - set(rule_lc.values()) - {t_all_txt}) | {fin_drop, fin_keep, t_fin} | bars["kept"] | exa["kept"]
    steps.append(final)
    save_steps(fig, steps, "pipeline_postprocessing_anim")
    plt.close(fig)


if __name__ == "__main__":
    slide_validation()
    r = Recording(rec)
    # the 120-s example window of the pose figures (pose frame = imaging frame, 20 Hz)
    t0, t1 = 6800, 9200
    slide_raw_data(r, t0, t1)
    slide_cells(r)
    slide_spikes(r, t0, t1)
    slide_postprocessing(r)
