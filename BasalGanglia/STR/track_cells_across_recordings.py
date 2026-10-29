"""Within-day cell tracking: match the cells extracted separately (pipeline_raw_spikes.py)
from the two recordings of a day (rec0 = 15-min drug/vehicle, rec1 = 45-min
amphetamine; same field of view, no remounting in between).

Per day:
  1. FOV shift rec0 -> rec1 by phase correlation of the two correlation images
  2. rec0 footprints (accepted cells, as in new_pipeline_outputs_*.npz) shifted by it
  3. footprint similarity = cosine similarity of the footprints for all pairs whose
     centroids lie within max_dist_px; one-to-one assignment (Hungarian) on similarity
  4. a pair is accepted if similarity >= sim_thr; sim_thr is set from a null
     distribution: the same assignment after displacing rec0 by null_shift_px in 4
     directions (neighbouring, non-identical cells), so that the false-match rate per
     rec0 cell is <= null_fpr

Outputs (cell_tracking/): per-day pairs (rec0 / rec1 cell index = row in the
recording's new_pipeline_outputs / spikes arrays), a per-day summary and a figure
(similarity real vs null, share of tracked cells per day, an example overlay).

The CNMF-E footprints are read once from the shared drive (hdf5) and cached locally.

  /home/rgast/conda/envs/parker/bin/python track_cells_across_recordings.py
"""

import os
import glob
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
from scipy.ndimage import shift as nd_shift
from scipy.optimize import linear_sum_assignment

results_root = "/mnt/kennedy_labdata/Parkerlab/results/raw_pipeline"     # CNMF-E hdf5 + QC table
local_root = "/home/rgast/data/parker_data/raw_pipeline_results"          # local mirror
out_dir = "/home/rgast/data/parker_data/cell_tracking"
tag = "p1_c0.7_pnr6"
max_dist_px = 6.0                 # max centroid distance of candidate pairs (~20 um)
null_shift_px = 20                # displacement for the null distribution
null_fpr = 0.02                   # accepted false-match rate per rec0 cell

def footprints(rec_dir):
    """Accepted cells' footprints (pixels x cells, F-order pixels, unit norm), dims,
    centroids (row, col) and correlation image, cached in the local mirror."""
    name = os.path.basename(rec_dir)
    cache = f"{local_root}/{name}/footprints_{tag}.npz"
    out = np.load(f"{local_root}/{name}/new_pipeline_outputs_{tag}.npz")
    if not os.path.isfile(cache):
        with h5py.File(f"{results_root}/{name}/cnmfe_{tag}.hdf5", "r") as h:
            g = h["estimates/A"]
            A = sp.csc_matrix((g["data"][...], g["indices"][...], g["indptr"][...]),
                              shape=tuple(g["shape"][...]))
            dims = tuple(int(d) for d in h["dims"][...])
        A = A[:, out["component"]]
        sp.save_npz(cache.replace(".npz", "_A.npz"), A)
        np.savez(cache, dims=dims)
    A = sp.load_npz(cache.replace(".npz", "_A.npz")).tocsc()
    dims = tuple(np.load(cache)["dims"])
    norm = np.sqrt(np.asarray(A.power(2).sum(0))).ravel() + 1e-12
    A = A @ sp.diags(1 / norm)
    return A, dims, out["centroids_px"], np.load(f"{local_root}/{name}/corr_image.npy")

def shift_footprints(A, dims, d):
    """Shift each footprint image by d = (rows, cols) pixels (bilinear)."""
    cols = []
    for k in range(A.shape[1]):
        img = A[:, k].toarray().reshape(dims, order="F")
        cols.append(sp.csc_matrix(nd_shift(img, d, order=1, cval=0).reshape(-1, 1, order="F")))
    B = sp.hstack(cols).tocsc()
    norm = np.sqrt(np.asarray(B.power(2).sum(0))).ravel() + 1e-12
    return B @ sp.diags(1 / norm)

def assign(A0, c0, A1, c1):
    """Hungarian assignment on footprint similarity among pairs within max_dist_px.
    Returns (i0, i1, similarity, distance) of all assigned candidate pairs."""
    S = (A0.T @ A1).toarray()
    D = np.linalg.norm(c0[:, None, :] - c1[None, :, :], axis=2)
    S[D > max_dist_px] = 0
    i, j = linear_sum_assignment(-S)
    ok = S[i, j] > 0
    return i[ok], j[ok], S[i, j][ok], D[i, j][ok]

def match_day(day):
    from skimage.registration import phase_cross_correlation
    A0, dims, c0, cn0 = footprints(f"{day}_rec0")
    A1, _, c1, cn1 = footprints(f"{day}_rec1")
    d, _, _ = phase_cross_correlation(cn1, cn0, upsample_factor=10, normalization=None)
    real = assign(shift_footprints(A0, dims, d), c0 + d, A1, c1)
    null = []
    for dn in [(null_shift_px, 0), (-null_shift_px, 0), (0, null_shift_px), (0, -null_shift_px)]:
        dd = d + np.array(dn)
        null.append(assign(shift_footprints(A0, dims, dd), c0 + dd, A1, c1)[2])
    return dict(day=os.path.basename(day), shift=d, n0=A0.shape[1], n1=A1.shape[1], real=real,
                null=null, A0=shift_footprints(A0, dims, d), A1=A1, dims=dims, cn1=cn1)

def main():
    os.makedirs(out_dir, exist_ok=True)
    days = sorted(p[:-5] for p in glob.glob(f"{local_root}/*_rec0")
                  if os.path.isdir(p[:-5] + "_rec1"))
    res = []
    for day in days:
        r = match_day(day)
        res.append(r)
        print(f"{r['day']:30s} shift {np.round(r['shift'], 1)}  n0 {r['n0']:4d}  n1 {r['n1']:4d}  "
              f"candidate pairs {len(r['real'][0])}")
    # threshold from the pooled null: false matches per rec0 cell <= null_fpr
    null_sims = np.concatenate([np.concatenate(r["null"]) for r in res])
    n_null_cells = sum(4 * r["n0"] for r in res)
    grid = np.linspace(0, 1, 201)
    fpr = np.array([(null_sims >= t).sum() / n_null_cells for t in grid])
    sim_thr = float(grid[np.argmax(fpr <= null_fpr)])
    print(f"\nsimilarity threshold {sim_thr:.3f} (null false matches per rec0 cell <= {null_fpr})")
    qc = pd.read_csv(f"{results_root}/qc_all_recordings.csv")
    rows = []
    for r in res:
        i0, i1, s, dist = r["real"]
        keep = s >= sim_thr
        pd.DataFrame(dict(rec0_cell=i0[keep], rec1_cell=i1[keep], similarity=s[keep],
                          dist_px=dist[keep])).to_csv(f"{out_dir}/{r['day']}_pairs.csv", index=False)
        q = qc[qc.session == r["day"]]
        rows.append(dict(day=r["day"], mouse=q.mouse.iloc[0] if len(q) else "",
                         spn=q.spn.iloc[0] if len(q) else "",
                         conditions=" / ".join(q.cond) if len(q) else "",
                         shift_row=r["shift"][0], shift_col=r["shift"][1], n_rec0=r["n0"],
                         n_rec1=r["n1"], n_tracked=int(keep.sum()),
                         frac_rec0_tracked=keep.sum() / max(r["n0"], 1),
                         frac_rec1_tracked=keep.sum() / max(r["n1"], 1),
                         median_similarity=float(np.median(s[keep])) if keep.any() else np.nan,
                         expected_false=float(np.interp(sim_thr, grid, fpr) * r["n0"])))
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{out_dir}/tracking_summary.csv", index=False)
    print(summ.drop(columns=["shift_row", "shift_col"]).round(3).to_string(index=False))
    make_figure(res, summ, null_sims, sim_thr)

def make_figure(res, summ, null_sims, sim_thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, blue, red = "#1f1f1e", "#5f5e5a", "#2a78d6", "#e34948"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")
    ax = axes[0]
    real_sims = np.concatenate([r["real"][2] for r in res])
    bins = np.linspace(0, 1, 51)
    ax.hist(real_sims, bins, color=blue, alpha=0.8, label="same position")
    ax.hist(null_sims, bins, weights=np.full(len(null_sims), 0.25), color=red, alpha=0.6,
            label=f"null: rec0 displaced {null_shift_px} px (per direction)")
    ax.axvline(sim_thr, color=ink, ls="--", lw=1)
    ax.text(sim_thr, ax.get_ylim()[1] * 0.95, f" threshold {sim_thr:.2f}", color=ink, fontsize=8, va="top")
    ax.set(xlabel="footprint similarity of assigned pair (cosine)", ylabel="# pairs",
           title="footprint similarity, all days")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    s = summ.sort_values("frac_rec0_tracked")
    y = np.arange(len(s))
    ax.barh(y, s.n_rec1, color="#c9c8c2", label="rec1 cells")
    ax.barh(y, s.n_rec0, color="#8fb8e8", label="rec0 cells")
    ax.barh(y, s.n_tracked, color=blue, label="tracked")
    ax.set_yticks(y, [f"{d} ({c.split(' / ')[0] if c else ''})" for d, c in zip(s.day, s.conditions)],
                  fontsize=6)
    ax.set(xlabel="# cells", title="cells per day: rec0, rec1, tracked")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax = axes[2]
    r = res[int(np.argmin(np.abs(summ.frac_rec0_tracked - summ.frac_rec0_tracked.median())))]
    i0, i1, sim, _ = r["real"]
    keep = sim >= sim_thr
    dims = r["dims"]
    ax.imshow(r["cn1"], cmap="gray", vmin=np.percentile(r["cn1"], 1), vmax=np.percentile(r["cn1"], 99.5))
    def contour(A, k, color, lw):
        img = A[:, k].toarray().reshape(dims, order="F")
        ax.contour(img, levels=[0.3 * img.max()], colors=[color], linewidths=lw)
    matched0, matched1 = set(i0[keep]), set(i1[keep])
    for k in range(r["A1"].shape[1]):
        contour(r["A1"], k, blue if k in matched1 else "#9ec5f0", 0.8)
    for k in range(r["A0"].shape[1]):
        contour(r["A0"], k, red if k in matched0 else "#f3a6a5", 0.8)
    ax.set_title(f"example (median day): {r['day']}\nrec1 (blue) and shifted rec0 (red) footprints; "
                 f"pale = untracked", fontsize=9)
    ax.axis("off")
    fig.suptitle("Within-day cell tracking from separately extracted recordings (footprint matching)",
                 fontsize=12, color=ink)
    fig.savefig(f"{out_dir}/cell_tracking_summary.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"saved {out_dir}/cell_tracking_summary.png")


if __name__ == "__main__":
    main()
