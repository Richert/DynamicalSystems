"""Visual comparison of old vs new cell extraction for one recording.

Shows example frames of the motion-corrected movie (as dF/F relative to the session
mean image, since the 1-photon background would otherwise hide the cells), taken at
times of high activity in the new pipeline, and marks the cells that are RECENTLY
ACTIVE at that time (calcium trace above act_frac x the neuron's maximum):
  - new cells (CNMF-E footprint contours, red)
  - old cells (circles at the old centroids mapped into pixels, blue)
  solid = matched across pipelines, dashed = unmatched.
An overview panel shows all cells on the correlation image. As in the pipeline's
evaluation, new cells pass CaImAn's quality check and the footprint shape filter, and
the old reference cells are those with clear activity (>= P.min_old_events events).

Also prints/saves diagnostics: where the unmatched old neurons lie (inside the field
of view? distance to the nearest new cell?) and the CNMF-E quality scores (SNR,
spatial consistency) of unmatched vs matched new cells.

Uses the outputs of pipeline_raw_spikes.py (memmap, CNMF-E model) and its matching.

  /home/rgast/conda/envs/parker/bin/python compare_cell_extraction.py \
      <drug>/<dose>/<session> [recording [work_root [min_corr min_pnr]]]
"""

import os
import sys
import json
import numpy as np
import pipeline_raw_spikes as P

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

n_examples = 4               # example frames
min_sep_s = 30.0             # min time between example frames (s)
avg_frames = 5               # frames averaged per example image (+/- avg_frames // 2)
act_frac = 0.3               # "recently active": trace > act_frac x the neuron's max
footprint_frac = 0.3         # contour level of a footprint (fraction of its max)
old_radius_px = 4.0          # circle radius for old cells (no footprints available)

# focus mode (flag "--focus"): poorly matched pairs at old events without a response
# of the matched new cell; zoomed event-triggered images show where fluorescence rose
focus = "--focus" in sys.argv
if focus:
    sys.argv.remove("--focus")
n_focus_pairs = 4            # pairs shown (lowest recall among old cells with many events)
min_focus_events = 15        # old events required for a pair to be shown
n_focus_events = 3           # missed old events shown per pair
crop_half_px = 20            # half-size of the zoomed crops
resp_frac = 0.2              # "no response": new trace stays < resp_frac x its max after the event

if len(sys.argv) > 1:
    P.raw_session = sys.argv[1]
    P.recording = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if len(sys.argv) > 3:
        P.work_root = sys.argv[3]
    if len(sys.argv) > 5:
        P.cnmf_params.update(min_corr=float(sys.argv[4]), min_pnr=float(sys.argv[5]))

# ---------------------------------------------------------------------------

def load():
    import glob
    import caiman as cm
    from caiman.source_extraction.cnmf import cnmf
    raw_file, old_file, cond, work = P.session_paths()
    fname = glob.glob(f"{work}/memmap_*_order_C_*.mmap")[0]
    Yr, dims, T = cm.load_memmap(fname)
    images = Yr.T.reshape((T,) + dims, order="F")
    cnm = cnmf.load_CNMF(f"{work}/cnmfe_{P.cnmf_tag()}.hdf5")
    est = cnm.estimates
    idx, _ = P.accepted_components(cnm, dims)       # quality check + shape filter
    A = est.A[:, idx].tocsc()
    C = est.C[idx]
    cents = cm.base.rois.com(A, *dims)
    snr = np.asarray(est.SNR_comp)[idx] if est.SNR_comp is not None else None
    rval = np.asarray(est.r_values)[idx] if est.r_values is not None else None
    cn = np.load(f"{work}/corr_image.npy")
    E_old, C_old, xy_old = P.load_old(old_file, cond)
    act = P.active_old(E_old)                        # reference: old cells with clear activity
    print(f"reference: {len(act)} of {len(E_old)} old cells with >= {P.min_old_events} events")
    load.yraw = C + est.YrA[idx]                     # raw extracted traces (for focus mode)
    load.old_index = act
    return work, images, dims, A, C, cents, snr, rval, cn, E_old[act], C_old[act], xy_old[act]

def old_to_px(xy_old, cents, pairs):
    """Affine map old um centroids -> new pixel (row, col), fitted on matched pairs."""
    X = np.column_stack([xy_old[[i for i, _ in pairs]], np.ones(len(pairs))])
    Y = cents[[j for _, j in pairs]]
    M, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return np.column_stack([xy_old, np.ones(len(xy_old))]) @ M

def pick_frames(active_new, T):
    """Frames with the most recently active new cells, >= min_sep_s apart."""
    score = active_new.sum(0).astype(float)
    k = avg_frames
    score = np.convolve(score, np.ones(k) / k, mode="same")
    chosen = []
    for t in np.argsort(score)[::-1]:
        if t < k or t > T - k:
            continue
        if all(abs(t - c) >= min_sep_s * P.fs for c in chosen):
            chosen.append(int(t))
        if len(chosen) == n_examples:
            break
    return sorted(chosen)

def footprint_img(A, k, dims):
    return A[:, k].toarray().reshape(dims, order="F")

def draw_cells(ax, A, dims, cents_old_px, new_ids, old_ids, matched_new, matched_old, lw=1.2):
    from matplotlib.patches import Circle
    new_c, old_c = "#ff4d4d", "#4da6ff"
    for k in new_ids:
        img = footprint_img(A, k, dims)
        ax.contour(img, levels=[footprint_frac * img.max()], colors=new_c, linewidths=lw,
                   linestyles="solid" if k in matched_new else "dashed")
    for i in old_ids:
        r, c = cents_old_px[i]
        ax.add_patch(Circle((c, r), old_radius_px, fill=False, ec=old_c, lw=lw,
                            ls="solid" if i in matched_old else "dashed"))

def focus_figure(work, images, dims, A, C, Yraw, cents, E_old, C_old, old_px, pairs, lag,
                 mean_img, old_index):
    """Poorly matched pairs: zoomed event-triggered images at old events where the
    matched new cell does not respond, with all new cells in the crop outlined."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    k = int(P.fs / P.fs_old)
    T = images.shape[0]
    H, W = dims
    um_per_px = 3.33
    # per-pair recall of old events by a new-trace response (trace-based, threshold-free)
    rows = []
    for i, j in pairs:
        f_ev = np.flatnonzero(E_old[i]) * k + lag
        f_ev = f_ev[(f_ev > 30) & (f_ev < T - 20)]
        if len(f_ev) < min_focus_events:
            continue
        resp = np.array([C[j, f:f + 12].max() - C[j, max(f - 4, 0):f].mean() for f in f_ev])
        missed = f_ev[resp < resp_frac * C[j].max()]
        rows.append((len(missed) / len(f_ev), i, j, missed))
    rows = sorted(rows, key=lambda r: -r[0])[:n_focus_pairs]
    if not rows:
        print("focus: no pair with enough events")
        return
    fig, axes = plt.subplots(len(rows), n_focus_events + 1, layout="constrained", squeeze=False,
                             figsize=(3.3 * n_focus_events + 6, 3.4 * len(rows)),
                             gridspec_kw=dict(width_ratios=[1] * n_focus_events + [2.2]))
    for r, (miss_frac, i, j, missed) in enumerate(rows):
        cy, cx = old_px[i]
        y0, y1 = int(max(cy - crop_half_px, 0)), int(min(cy + crop_half_px, H))
        x0, x1 = int(max(cx - crop_half_px, 0)), int(min(cx + crop_half_px, W))
        near = [m for m in range(len(C)) if y0 <= cents[m, 0] < y1 and x0 <= cents[m, 1] < x1]
        sel = missed[np.linspace(0, len(missed) - 1, min(n_focus_events, len(missed))).astype(int)]
        for c in range(n_focus_events):
            ax = axes[r, c]
            if c >= len(sel):
                ax.axis("off")
                continue
            f = int(sel[c])
            post = images[f:f + 8].mean(0)
            pre = images[max(f - 25, 0):f - 5].mean(0)
            d = ((post - pre) / (np.abs(mean_img) + 1e-6))[y0:y1, x0:x1]
            lim = np.percentile(np.abs(d), 99.5)
            ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=(x0, x1, y1, y0))
            # which new cell in the crop rises most (mean event-triggered change on its footprint)
            best, best_v = None, 0.0
            for m in near:
                a = A[:, m].toarray().reshape(dims, order="F")
                w = a / a.sum()
                v = float((w * (post - pre) / (np.abs(mean_img) + 1e-6)).sum())
                lw, col = (2.0, "k") if m == j else (0.8, "0.25")
                ax.contour(np.arange(W), np.arange(H), a, levels=[0.3 * a.max()], colors=col,
                           linewidths=lw)
                if v > best_v:
                    best, best_v = m, v
            for io in range(len(old_px)):
                oy, ox = old_px[io]
                if y0 <= oy < y1 and x0 <= ox < x1:
                    ax.add_patch(Circle((ox, oy), 4, fill=False, ec="#1f77b4",
                                        lw=2.0 if io == i else 0.8, ls="-" if io == i else "--"))
            ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
            ax.set_xticks([]); ax.set_yticks([])
            if best is None:
                who = "none"
            elif best == j:
                who = "matched new cell"
            else:
                who = f"new {best} ({np.linalg.norm(cents[best] - cents[j]) * um_per_px:.0f} um away)"
            ax.set_title(f"old event t={f / P.fs:.1f}s\nlargest rise: {who}", fontsize=8)
        # traces around the shown events
        ax = axes[r, -1]
        f_mid = int(sel[0])
        w0, w1 = max(f_mid - 200, 0), min(f_mid + 400, T)
        t = np.arange(w0, w1) / P.fs
        sc = C[j].max() + 1e-12
        ax.plot(t, Yraw[j, w0:w1] / sc, c="0.7", lw=0.5, label="new raw (C + residual)")
        ax.plot(t, C[j, w0:w1] / sc, c="#d1495b", lw=0.9, label="new denoised")
        i5 = (np.arange(w0, w1) - lag) // k
        i5 = np.clip(i5, 0, C_old.shape[1] - 1)
        ax.plot(t, C_old[i, i5] / (C_old[i].max() + 1e-12) + 1.2, c="#3b6fb6", lw=0.9,
                label="old trace")
        ev = np.flatnonzero(E_old[i]) * k + lag
        ev = ev[(ev >= w0) & (ev < w1)]
        ax.plot(ev / P.fs, np.full(len(ev), 2.3), "|", c="#3b6fb6", ms=8)
        ax.axvline(f_mid / P.fs, c="k", ls=":", lw=0.8)       # first shown event
        ax.set(yticks=[], xlabel=f"time (s), around the first shown event", xlim=(w0 / P.fs, w1 / P.fs),
               title=f"old {int(old_index[i])} / new {j}: {miss_frac:.0%} of old events "
                     f"without a new response", )
        ax.title.set_fontsize(9)
        if r == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{P.raw_session} rec{P.recording}: poorly matched pairs - event-triggered "
                 f"change (0-0.4 s after vs 1.25-0.25 s before each old event, / mean image); "
                 f"bold black = matched new cell, grey = other new cells, blue = old cells",
                 fontsize=10)
    out = f"{work}/cell_focus_{P.cnmf_tag()}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    work, images, dims, A, C, cents, snr, rval, cn, E_old, C_old, xy_old = load()
    T = images.shape[0]

    # matching exactly as in the pipeline
    lag = P.estimate_lag(C, C_old)
    C5 = P.to_old_rate(C, lag)
    T5 = min(C5.shape[1], C_old.shape[1])
    R = P.corr_matrix(C_old[:, :T5], C5[:, :T5])
    pairs, _ = P.match_neurons(R, cents, xy_old)
    matched_new = {j for _, j in pairs}
    matched_old = {i for i, _ in pairs}
    old_px = old_to_px(xy_old, cents, pairs)

    # recently active cells per frame (new at 20 Hz; old mapped from 5 Hz)
    active_new = C > act_frac * C.max(1, keepdims=True)
    act_old5 = C_old > act_frac * np.maximum(C_old.max(1, keepdims=True), 1e-12)
    old_idx = np.clip((np.arange(T) - lag) // int(P.fs / P.fs_old), 0, C_old.shape[1] - 1)
    frames = pick_frames(active_new, T)

    # dF/F images relative to the session mean image
    mean_img = images[::max(1, T // 2000)].mean(0)
    if focus:
        focus_figure(work, images, dims, A, C, load.yraw, cents, E_old, C_old, old_px, pairs,
                     lag, mean_img, load.old_index)
        return

    # --- diagnostics ---------------------------------------------------------
    H, W = dims
    unm_old = [i for i in range(len(xy_old)) if i not in matched_old]
    inside = [(0 <= old_px[i, 0] < H) and (0 <= old_px[i, 1] < W) for i in unm_old]
    d_near = [np.min(np.linalg.norm(cents - old_px[i], axis=1)) for i in unm_old]
    n_ev = E_old.sum(1)
    diag = dict(
        dims=list(dims), n_old=len(xy_old), n_new=len(C), n_matched=len(pairs),
        unmatched_old=len(unm_old),
        unmatched_old_inside_fov=float(np.mean(inside)) if unm_old else np.nan,
        unmatched_old_dist_to_nearest_new_px_median=float(np.median(d_near)) if unm_old else np.nan,
        unmatched_old_with_new_cell_within_2r=float(np.mean(np.array(d_near) < 2 * old_radius_px)) if unm_old else np.nan,
        unmatched_old_events_median=float(np.median(n_ev[unm_old])) if unm_old else np.nan,
        matched_old_events_median=float(np.median(n_ev[list(matched_old)])) if pairs else np.nan,
    )
    unm_new = [k for k in range(len(C)) if k not in matched_new]
    if snr is not None:
        diag.update(snr_matched_median=float(np.median(snr[list(matched_new)])),
                    snr_unmatched_median=float(np.median(snr[unm_new])) if unm_new else np.nan,
                    rval_matched_median=float(np.median(rval[list(matched_new)])),
                    rval_unmatched_median=float(np.median(rval[unm_new])) if unm_new else np.nan)
    print(json.dumps(diag, indent=1))
    json.dump(diag, open(f"{work}/cell_overlay_diagnostics_{P.cnmf_tag()}.json", "w"), indent=1)

    # --- figure ------------------------------------------------------------
    ncol = 3
    nrow = int(np.ceil((len(frames) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.2 * nrow * H / W + 0.6),
                             layout="constrained", squeeze=False)
    ax = axes.flat[0]
    ax.imshow(cn, cmap="gray")
    draw_cells(ax, A, dims, old_px, range(len(C)), range(len(xy_old)), matched_new,
               matched_old, lw=0.7)
    ax.set_title(f"all cells on correlation image\nold {len(xy_old)} / new {len(C)} / "
                 f"matched {len(pairs)}", fontsize=10)
    for ax, t in zip(axes.flat[1:], frames):
        sl = slice(t - avg_frames // 2, t + avg_frames // 2 + 1)
        img = images[sl].mean(0)
        dff = (img - mean_img) / (np.abs(mean_img) + 1e-6)
        lo, hi = np.percentile(dff, [1, 99.8])
        ax.imshow(dff, cmap="gray", vmin=lo, vmax=hi)
        new_ids = np.flatnonzero(active_new[:, t])
        old_ids = np.flatnonzero(act_old5[:, old_idx[t]])
        draw_cells(ax, A, dims, old_px, new_ids, old_ids, matched_new, matched_old)
        n_both = len({j for i, j in pairs if i in set(old_ids) and j in set(new_ids)})
        ax.set_title(f"t = {t / P.fs:.1f} s: active new {len(new_ids)} / old {len(old_ids)} "
                     f"(both, matched {n_both})", fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(frames) + 1:]:
        ax.axis("off")
    handles = [Line2D([], [], c="#ff4d4d", ls="-", label="new, matched"),
               Line2D([], [], c="#ff4d4d", ls="--", label="new, unmatched"),
               Line2D([], [], c="#4da6ff", ls="-", label="old, matched"),
               Line2D([], [], c="#4da6ff", ls="--", label="old, unmatched")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle(f"{P.raw_session} rec{P.recording} ({P.cnmf_tag()}): recently active cells "
                 f"(trace > {act_frac:g} x max) at high-activity times; images = dF/F "
                 f"(mean of {avg_frames} frames)", fontsize=11)
    out = f"{work}/cell_overlay_{P.cnmf_tag()}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
