"""Behaviour-independent SPN activity as a function of drug condition (raw pipeline).

Uses the outputs of pipeline_raw_spikes.py (CaImAn CNMF-E + OASIS events at 20 Hz).
For every recording, M random windows of length T are drawn (several T), each with a
fixed number of randomly sampled cells, and per window:
  from the inferred spike (event) times:
  mean_log, sd_log  mean / SD over the sampled cells of log10 event rate (Hz); cells
                    without events sit at a floor of half an event per window
  isi_med_log       median inter-spike interval (log10 s) of all ISIs in the window
                    (ISIs within a window, so capped at T)
  isi_burst         share of ISIs within burst_isi_s = [0.2, 0.5) s (shorter ISIs,
                    1-3 frames, are mostly one rising edge split by the deconvolution)
  from CNMF-E's denoised calcium traces C (active cells = >= 1 event in the window):
  corr_active       mean pairwise Pearson correlation, pairs further apart than
                    min_pair_dist_um
  corr_excess       corr_active minus the same for circularly shifted (shuffled) traces
  pr_ratio          participation ratio divided by that of the shuffled traces (plain
                    PR ~ number of active cells)
Recordings are summarised by the median over windows; conditions are compared per SPN
type with value ~ C(condition) + (1|mouse) and Wald contrasts (BH-FDR).
The pooled per-cell log rates give the rate distribution per condition: silent share
+ Gaussian mixture (1..3 components, BIC) on the active cells.

Conditions (per day: 15-min recording -> amphetamine -> 10-min gap -> 45-min recording):
  Vehicle day rec0 = Control, rec1 = Amphetamine; HighDose day rec0 = <drug>, rec1 = A + <drug>

Run with the arhmm env:
  /home/rgast/conda/envs/arhmm/bin/python analysis_activity_by_condition.py
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import analysis_arhmm_spn_states as A   # SPN type of each mouse, mixed-model helpers

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

results_root = "/mnt/kennedy_labdata/Parkerlab/results"
# local mirror of the pipeline outputs (traces, spikes, tables; no movies) -- reading the
# shared drive was the bottleneck; refresh with rsync from {results_root}/raw_pipeline
pipe_dir = "/home/rgast/data/parker_data/raw_pipeline_results"
out_dir = "/home/rgast/data/parker_data/behavior_independent"
drug = "Clozapine"
tag = "p1_c0.7_pnr6"                     # CNMF-E setting of the pipeline outputs
spike_thr = 1.5
fs = 20.0

window_lengths_s = [10, 30, 50]          # T
n_windows = 50                           # M per recording
n_cells = 50                             # cells sampled per window
min_cells = 50                           # recordings with fewer accepted cells are dropped
exclude_mice_with_small_recordings = False   # True: drop the whole mouse instead
min_active = 10                          # active cells needed for corr / PR in a window
burst_isi_s = (0.2, 0.5)                 # burst ISIs (as in the pipeline evaluation)
isi_bins = np.logspace(np.log10(0.05), np.log10(60), 45)   # pooled ISI histograms (s)
min_pair_dist_um = 20.0                  # skip nearby pairs (1-photon crosstalk)
um_per_px = 3.33
seed = 1

conditions = ["Control", drug, "Amphetamine", f"A + {drug}"]
readouts = ["mean_log", "sd_log", "isi_med_log", "isi_burst", "corr_active", "corr_excess",
            "pr_ratio"]
contrasts = [(drug, "Control"), ("Amphetamine", "Control"), (f"A + {drug}", "Control"),
             (f"A + {drug}", "Amphetamine"), (f"A + {drug}", drug)]

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def recordings():
    """One row per recording: paths, mouse, SPN type, condition, cell count."""
    rows = []
    for d in sorted(glob.glob(f"{pipe_dir}/*_rec[01]")):
        name = os.path.basename(d)
        _, mouse, day, *_ = name.split("_")
        mouse = {"m85": "m085"}.get(mouse, mouse)          # raw folder naming quirk
        rec = int(name[-1])
        dose_day = "HighDose" if day.endswith("04") else "Vehicle"
        cond = {("Vehicle", 0): "Control", ("Vehicle", 1): "Amphetamine",
                ("HighDose", 0): drug, ("HighDose", 1): f"A + {drug}"}[(dose_day, rec)]
        out = np.load(f"{d}/new_pipeline_outputs_{tag}.npz")
        rows.append(dict(dir=d, recording=name, mouse=mouse, spn=A.spn_type(mouse),
                         condition=cond, n_cells=out["C"].shape[0]))
    df = pd.DataFrame(rows)
    small = df.n_cells < min_cells
    if exclude_mice_with_small_recordings:
        df = df[~df.mouse.isin(df[small].mouse)]
    else:
        df = df[~small]
    return df

# ---------------------------------------------------------------------------
# window readouts
# ---------------------------------------------------------------------------

def pr(X):
    ev = np.linalg.eigvalsh(np.cov(X))
    ev = ev[ev > 0]
    return ev.sum() ** 2 / (ev ** 2).sum() if len(ev) else np.nan

def mean_pair_corr(X, far):
    Xc = X - X.mean(1, keepdims=True)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    R = Xc @ Xc.T
    iu = np.triu_indices(len(X), 1)
    keep = far[iu]
    return R[iu][keep].mean() if keep.any() else np.nan

def window_readouts(spk, C, cents, T_s, rng):
    """M windows of T_s seconds: per-window readouts (rates and ISIs from the event
    trains spk, correlations / PR from the denoised traces C), all per-cell event
    counts and all ISIs (s)."""
    n_frames = int(T_s * fs)
    floor = 0.5 / T_s                                          # half an event per window
    rows, all_counts, all_isis = [], [], []
    for _ in range(n_windows):
        t0 = rng.integers(0, spk.shape[1] - n_frames)
        cells = rng.choice(spk.shape[0], n_cells, replace=False)
        frames = np.arange(t0, t0 + n_frames)
        W = spk[np.ix_(cells, frames)]
        counts = W.sum(1)
        x = np.log10(np.maximum(counts / T_s, floor))
        all_counts.append(counts)
        isis = np.concatenate([np.diff(np.flatnonzero(w)) / fs for w in W if w.sum() > 1]
                              or [np.array([])])
        all_isis.append(isis)
        row = dict(mean_log=x.mean(), sd_log=x.std(), frac_active=(counts > 0).mean(),
                   isi_med_log=np.log10(np.median(isis)) if len(isis) >= 5 else np.nan,
                   isi_burst=np.mean((isis >= burst_isi_s[0]) & (isis < burst_isi_s[1]))
                   if len(isis) >= 5 else np.nan,
                   corr_active=np.nan, corr_excess=np.nan, pr_ratio=np.nan)
        act = np.flatnonzero(counts > 0)
        if len(act) >= min_active:
            X = C[np.ix_(cells[act], frames)].astype(float)
            varying = X.std(1) > 0
            X, c = X[varying], cents[cells[act]][varying]
            if len(X) >= min_active:
                far = np.linalg.norm(c[:, None] - c[None], axis=2) * um_per_px >= min_pair_dist_um
                Xs = np.array([np.roll(v, rng.integers(1, X.shape[1])) for v in X])
                cr, cs = mean_pair_corr(X, far), mean_pair_corr(Xs, far)
                row.update(corr_active=cr, corr_excess=cr - cs, pr_ratio=pr(X) / pr(Xs))
        rows.append(row)
    return pd.DataFrame(rows), np.concatenate(all_counts), np.concatenate(all_isis)

# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def condition_stats(rec_df, q):
    """value ~ C(condition) + (1|mouse) per SPN type; Wald contrasts, BH-FDR."""
    import statsmodels.api as sm
    from patsy import dmatrices
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    out = []
    for spn, d in rec_df.dropna(subset=[q]).groupby("spn"):
        d = d.copy()
        d["condition"] = pd.Categorical(d.condition, categories=[c for c in conditions
                                                                 if c in set(d.condition)])
        y, X = dmatrices(f"{q} ~ C(condition)", d, return_type="dataframe")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = A._fit_mixedlm(y, X, d["mouse"].values)
        if not A._valid_fit(m, "mixedlm", d.mouse.nunique())[0]:
            m = sm.OLS(y, X).fit(cov_type="cluster",
                                 cov_kwds={"groups": d.mouse.astype("category").cat.codes})
        p, cov = m.params, m.cov_params()        # not "C": patsy's C() is resolved locally
        def vec(c):
            v = pd.Series(0.0, index=p.index)
            v["Intercept"] = 1.0
            if c != d.condition.cat.categories[0]:
                v[f"C(condition)[T.{c}]"] = 1.0
            return v
        rows = []
        for a, b in contrasts:
            if a not in d.condition.cat.categories or b not in d.condition.cat.categories:
                continue
            v = vec(a) - vec(b)
            est = float(v @ p)
            se = float(np.sqrt(v @ cov.loc[p.index, p.index] @ v))
            rows.append(dict(spn=spn, quantity=q, contrast=f"{a} - {b}", diff=est, se=se,
                             p=2 * norm.sf(abs(est / se)), n_mice=d.mouse.nunique()))
        if rows:
            r = pd.DataFrame(rows)
            r["q_fdr"] = multipletests(r.p, method="fdr_bh")[1]
            out.append(r)
    return pd.concat(out) if out else pd.DataFrame()

# ---------------------------------------------------------------------------
# rate distributions
# ---------------------------------------------------------------------------

def fit_distribution(counts, T_s, rng):
    """Silent share + GMM (1..3 components, BIC) on the active cells' log rates.
    Event counts are integers, so log rates pile up at log10(1/T), log10(2/T), ...;
    for the fit they are dequantised (count + U(0,1)) to describe the underlying
    continuous rate distribution instead of the count grid."""
    from sklearn.mixture import GaussianMixture
    c = counts[counts > 0]
    act = np.log10((c + rng.random(len(c))) / T_s)
    sub = rng.choice(act, min(len(act), 20000), replace=False).reshape(-1, 1)
    fits = [GaussianMixture(k, random_state=seed).fit(sub) for k in (1, 2, 3)]
    best = min(fits, key=lambda g: g.bic(sub))
    return dict(silent=float(np.mean(counts == 0)), gmm=best, active=act)

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    recs = recordings()
    print(f"{len(recs)} recordings ({recs.mouse.nunique()} mice); "
          f"{'per-mouse' if exclude_mice_with_small_recordings else 'per-recording'} rule")
    print(recs.groupby(["spn", "condition"]).mouse.nunique().unstack()[conditions].to_string())
    suffix = "_permouse" if exclude_mice_with_small_recordings else ""

    # each recording is loaded once; all window lengths are evaluated on it
    rec_rows, pooled, isi_hist = [], {}, {}
    for n, (_, r) in enumerate(recs.iterrows(), start=1):
        spk = np.load(f"{r.dir}/spikes_{tag}_thr{spike_thr}.npy")
        out = np.load(f"{r.dir}/new_pipeline_outputs_{tag}.npz")
        C, cents = out["C"].astype(np.float32), out["centroids_px"]
        for T_s in window_lengths_s:
            win, counts, isis = window_readouts(spk, C, cents, T_s, rng)
            rec_rows.append(dict(recording=r.recording, mouse=r.mouse, spn=r.spn,
                                 condition=r.condition, T_s=T_s,
                                 **win.median(numeric_only=True).to_dict()))
            key = (T_s, r.spn, r.condition)
            pooled.setdefault(key, []).append(counts)
            # ISIs are multiples of 1/fs: dequantise (+/- half a frame) for the histogram,
            # otherwise log bins narrower than 50 ms alternate between full and empty
            jit = isis + (rng.random(len(isis)) - 0.5) / fs
            isi_hist[key] = isi_hist.get(key, 0) + np.histogram(jit, bins=isi_bins)[0]
        print(f"  {n}/{len(recs)} {r.recording}")
    rec_df = pd.DataFrame(rec_rows)
    all_stats = []
    for T_s in window_lengths_s:
        for q in readouts:
            s = condition_stats(rec_df[rec_df.T_s == T_s], q)
            if len(s):
                all_stats.append(s.assign(T_s=T_s))
    stats = pd.concat(all_stats)
    dists = {key: fit_distribution(np.concatenate(v), key[0], rng) for key, v in pooled.items()}
    # pooled ISI densities (per log10 unit), per window length / SPN type / condition
    widths = np.diff(np.log10(isi_bins))
    centers = np.sqrt(isi_bins[1:] * isi_bins[:-1])
    pd.DataFrame([dict(T_s=k[0], spn=k[1], condition=k[2], isi_s=c,
                       density=h / max(hist.sum(), 1) / w, n_isi=int(hist.sum()))
                  for k, hist in isi_hist.items() for c, h, w in zip(centers, hist, widths)
                  ]).to_csv(f"{out_dir}/isi_distributions{suffix}.csv", index=False)
    rec_df.to_csv(f"{out_dir}/recording_readouts{suffix}.csv", index=False)
    stats.to_csv(f"{out_dir}/condition_contrasts{suffix}.csv", index=False)
    dist_rows = []
    for (T_s, spn, cond), d in dists.items():
        g = d["gmm"]
        order = np.argsort(g.means_.ravel())
        dist_rows.append(dict(T_s=T_s, spn=spn, condition=cond, silent=d["silent"],
                              n_components=g.n_components,
                              means=np.round(g.means_.ravel()[order], 3).tolist(),
                              sds=np.round(np.sqrt(g.covariances_.ravel()[order]), 3).tolist(),
                              weights=np.round(g.weights_[order], 3).tolist()))
    pd.DataFrame(dist_rows).to_csv(f"{out_dir}/rate_distributions{suffix}.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nper-condition means of recording medians (T = 30 s):")
    print(rec_df[rec_df.T_s == 30].groupby(["spn", "condition"])[readouts + ["frac_active"]]
          .mean().reindex(conditions, level=1).round(3).to_string())
    print("\nsignificant contrasts (q < 0.05):")
    sig = stats[stats.q_fdr < 0.05]
    print(sig[["T_s", "spn", "quantity", "contrast", "diff", "q_fdr", "n_mice"]]
          .round(4).to_string(index=False) if len(sig) else "  none")
    print(f"saved result tables to {out_dir}/ (figures: plotting_activity_by_condition.py)")

if __name__ == "__main__":
    main()
