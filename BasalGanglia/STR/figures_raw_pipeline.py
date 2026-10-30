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


if __name__ == "__main__":
    r = Recording(rec)
    # the 120-s example window of the pose figures (pose frame = imaging frame, 20 Hz)
    t0, t1 = 6800, 9200
    slide_raw_data(r, t0, t1)
