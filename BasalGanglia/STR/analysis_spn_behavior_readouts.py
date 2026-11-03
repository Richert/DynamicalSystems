"""SPN population readouts per behaviour label for Control, Amphetamine and A + Drug (dose set by
A_DRUG: "low" / "high"), D1 / D2 / combined, from the SPN data file (build_spn_dataset.py).

Equal-time / equal-neuron sampling: per label a time budget T (smallest median label time over
the conditions, clipped to [t_min, t_max]); recordings with less than T in the label are left
out. Per recording and label, n_rep random draws of n_sub cells and of bouts (random order,
the last one cut) up to T; readouts averaged over the draws.
  log rate      per cell log10(events / T), zero counts floored at 0.5 events; per draw the
                mean and SD over the drawn cells
  within-bout   mean pairwise correlation of the cells' spike trains, smoothed with a Gaussian
  correlation   kernel (sigma kernel_s, spike-time jitter), within bouts >= corr_bout_s drawn up
                to the budget T_corr (as T, from the time in bouts >= corr_bout_s); pairs of
                cells with spikes in (or within 3 sigma before) the bout
  participation participation ratio (sum lambda)^2 / sum lambda^2 of the covariance of the
  ratio         smoothed spike trains of the drawn cells over the drawn (concatenated) bouts:
                high when different cells carry the activity in different bouts
Rate distributions: per cell log10(mean event count over the draws / T) (all cells, equal time),
pooled over the recordings of a condition.

Outputs (OUT_DIR): readouts_<A_DRUG>.csv, cell_log_rates.csv.gz, one figure per label.

  conda run -n parker python analysis_spn_behavior_readouts.py
"""
import os
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter1d

dataset = os.environ.get("DATASET", "/home/rgast/data/parker_data/dataset/clozapine_spn.h5")
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/spn_behavior_states")
a_drug = os.environ.get("A_DRUG", "high")
fs = 20.0
conditions = ["Control", "Amphetamine", f"A + Clozapine {a_drug}"]
cond_short = {"Control": "Control", "Amphetamine": "Amph.", f"A + Clozapine {a_drug}": f"Amph. + Cloz. {a_drug}"}
min_bouts = 10
kernel_s = 0.1                     # Gaussian kernel SD for the correlation measures
corr_bout_s = 0.5                  # bouts used for the within-bout correlation
t_min, t_max = 20.0, 120.0         # time budget per label (rates, participation ratio)
tc_min, tc_max = 10.0, 60.0        # time budget per label (within-bout correlation)
n_sub, n_rep = 50, 50              # cells per draw, draws
seed = 1

def bouts_of(lab):
    st = np.flatnonzero(np.diff(np.r_[-2, lab]) != 0)
    en = np.r_[st[1:], len(lab)]
    return st, en, lab[st]

def mean_offdiag(R):
    n = len(R)
    return np.nan if n < 2 else (np.nansum(R) - np.nansum(np.diag(R))) / (n * n - n)

def label_bouts(lab, names):
    st, en, bl = bouts_of(lab)
    return {name: (st[bl == k], en[bl == k]) for k, name in enumerate(names)}

def draw_bouts(st, en, budget, rng):
    """Bouts in random order up to budget frames (last one cut): list of (a, b)."""
    out, tot = [], 0
    for i in rng.permutation(len(st)):
        a, b = st[i], min(en[i], st[i] + budget - tot)
        out.append((a, b))
        tot += b - a
        if tot >= budget:
            return out
    return None

def participation_ratio(X):
    X = X - X.mean(1, keepdims=True)
    C = X @ X.T
    tr = np.trace(C)
    return np.nan if tr <= 0 else tr ** 2 / np.sum(C * C)

def budgets(h):
    """Time budgets (s) per label: smallest median over the conditions, clipped."""
    rows = []
    for day in h:
        for rec in ("rec0", "rec1"):
            r = h[day][rec]
            if r.attrs["condition"] not in conditions or not r["behaviour"].attrs.get("available", False):
                continue
            lab = r["behaviour/label_clean"][:]
            for name, (st, en) in label_bouts(lab, list(r["behaviour"].attrs["label_names"])).items():
                L = (en - st) / fs
                rows.append(dict(condition=r.attrs["condition"], label=name, secs=L.sum(),
                                 secs_long=L[L >= corr_bout_s].sum()))
    d = pd.DataFrame(rows).groupby(["label", "condition"]).median().groupby("label").min()
    return (d.secs.clip(t_min, t_max).to_dict(), d.secs_long.clip(tc_min, tc_max).to_dict())

def recording_readouts(r, attrs, rng, T_lab, Tc_lab):
    beh = r["behaviour"]
    names = list(beh.attrs["label_names"])
    lab = beh["label_clean"][:]
    T = len(lab)
    ptr, fr = r["event_ptr"][:], r["event_frames"][:]
    n = len(ptr) - 1
    E = np.zeros((n, T), np.float32)
    for i in range(n):
        E[i, fr[ptr[i]:ptr[i + 1]]] = 1
    Sm = gaussian_filter1d(E, kernel_s * fs, axis=1, mode="constant")
    cs = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(E, 1)], 1)   # event counts
    pad = int(3 * kernel_s * fs)
    rows, cells = [], []
    for name, (st, en) in label_bouts(lab, names).items():
        if name not in T_lab:
            continue
        budget = int(round(T_lab[name] * fs))
        secs = (en - st).sum() / fs
        if len(st) < min_bouts or (en - st).sum() < budget:
            continue
        long_ = (en - st) >= corr_bout_s * fs
        bc = int(round(Tc_lab[name] * fs))
        do_corr = (en - st)[long_].sum() >= bc
        m_lr, s_lr, sil, pr, wb = [], [], [], [], []
        cnt_sum = np.zeros(n)
        for _ in range(n_rep):
            sel = np.sort(rng.choice(n, min(n_sub, n), replace=False))
            bts = draw_bouts(st, en, budget, rng)
            cnt = sum(cs[:, b] - cs[:, a] for a, b in bts)
            cnt_sum += cnt
            lr = np.log10(np.maximum(cnt[sel], 0.5) / T_lab[name])
            m_lr.append(lr.mean()); s_lr.append(lr.std()); sil.append(np.mean(cnt[sel] == 0))
            fr_idx = np.concatenate([np.arange(a, b) for a, b in bts])
            pr.append(participation_ratio(Sm[np.ix_(sel, fr_idx)]))
            if do_corr:
                v = []
                for a, b in draw_bouts(st[long_], en[long_], bc, rng):
                    if (b - a) < corr_bout_s * fs:
                        continue
                    act = sel[E[sel, max(0, a - pad):b].sum(1) > 0]
                    X = Sm[act, a:b]
                    X = X - X.mean(1, keepdims=True)
                    sd = np.sqrt((X ** 2).sum(1))
                    if (sd > 0).sum() < 2:
                        continue
                    Z = X[sd > 0] / sd[sd > 0, None]
                    v.append(mean_offdiag(Z @ Z.T))
                wb.append(np.mean(v) if v else np.nan)
        base = dict(day=attrs["day"], condition=attrs["condition"], mouse=attrs["mouse"],
                    spn=attrs["spn"], label=name)
        rows.append(dict(base, n_cells=n, seconds=secs, bouts=len(st), T=T_lab[name],
                         T_corr=Tc_lab[name] if do_corr else np.nan,
                         mean_log_rate=np.mean(m_lr), sd_log_rate=np.mean(s_lr), silent=np.mean(sil),
                         within_bout_corr=np.nanmean(wb) if do_corr and np.isfinite(wb).any() else np.nan,
                         participation_ratio=np.nanmean(pr)))
        mc = cnt_sum / n_rep
        cells.append(pd.DataFrame(dict(base, log_rate=np.log10(np.maximum(mc, 0.5) / T_lab[name]),
                                       silent=mc == 0)))
    return rows, cells

def figure_label(rd, cl, label, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cols = dict(zip(conditions, ["#2a78d6", "#eda100", "#e34948"]))
    groups = [("D1", ["D1"]), ("D2", ["D2"]), ("D1 + D2", ["D1", "D2"])]
    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.45, wspace=0.35)
    bins = np.linspace(-2.75, 0.25, 31)
    for j, (gname, spns) in enumerate(groups):
        ax = fig.add_subplot(gs[0, j])
        for c in conditions:
            d = cl[(cl.label == label) & (cl.condition == c) & cl.spn.isin(spns)]
            if not len(d):
                continue
            h, _ = np.histogram(d.log_rate.clip(bins[0], bins[-1]), bins=bins)
            ax.stairs(h / len(d), bins, color=cols[c], lw=1.6,
                      label=f"{cond_short[c]} (n = {len(d)}, silent {rd[(rd.label == label) & (rd.condition == c) & rd.spn.isin(spns)].silent.mean():.0%})")
        ax.set(xlabel="log10 event rate (Hz)", ylabel="share of cells" if j == 0 else None)
        ax.set_title(f"{gname}: rate distribution", loc="left", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    readouts = [("mean_log_rate", "mean log10 rate"), ("sd_log_rate", "SD of log10 rate"),
                ("within_bout_corr", "within-bout pairwise corr."), ("participation_ratio", f"participation ratio ({n_sub} cells)")]
    for i, (col, lab) in enumerate(readouts):
        ax = fig.add_subplot(gs[1, i])
        for gi, (gname, spns) in enumerate(groups):
            for ci, c in enumerate(conditions):
                v = rd[(rd.label == label) & (rd.condition == c) & rd.spn.isin(spns)][col].dropna().values
                x = gi + (ci - 1) * 0.25
                if len(v):
                    ax.scatter(x + np.random.default_rng(ci).uniform(-0.06, 0.06, len(v)), v, s=10,
                               color=cols[c], alpha=0.45, lw=0)
                    ax.errorbar(x, v.mean(), v.std() / np.sqrt(len(v)), color=cols[c], marker="o",
                                ms=5, capsize=2, lw=1.4)
        ax.set_xticks(range(3), [g for g, _ in groups])
        ax.set_title(lab, loc="left", fontsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.legend(handles=[Line2D([], [], color=cols[c], marker="o", lw=1.4, label=cond_short[c]) for c in conditions],
               loc="lower center", ncol=3, frameon=False)
    r0 = rd[rd.label == label]
    fig.suptitle(f"SPN activity during '{label}' (recordings: dots, mean ± SEM). Equal sampling: "
                 f"{r0['T'].iloc[0]:.0f} s and {n_sub} cells per draw, {n_rep} draws; correlations: "
                 f"Gaussian kernel {kernel_s * 1000:.0f} ms, bouts >= {corr_bout_s:g} s "
                 f"({r0.T_corr.max():.0f} s per draw)", y=0.99, fontsize=10)
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)

def main():
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, cells = [], []
    with h5py.File(dataset, "r") as h:
        T_lab, Tc_lab = budgets(h)
        print("budgets (s):", {k: round(v) for k, v in T_lab.items()}, {k: round(v) for k, v in Tc_lab.items()})
        for day in h:
            g = h[day]
            for rec in ("rec0", "rec1"):
                r = g[rec]
                if r.attrs["condition"] not in conditions or not r["behaviour"].attrs.get("available", False):
                    continue
                attrs = dict(g.attrs, day=day, condition=r.attrs["condition"])
                a, b = recording_readouts(r, attrs, rng, T_lab, Tc_lab)
                rows += a
                cells += b
            print(day, flush=True)
    rd = pd.DataFrame(rows)
    cl = pd.concat(cells, ignore_index=True)
    rd.to_csv(f"{out_dir}/readouts_{a_drug}.csv", index=False)
    cl.to_csv(f"{out_dir}/cell_log_rates_{a_drug}.csv.gz", index=False)
    labs = rd.groupby("label").condition.nunique()
    for label in labs.index:
        figure_label(rd, cl, label, f"{out_dir}/readouts_{a_drug}_{label.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    pd.set_option("display.width", 220)
    print(rd.groupby(["label", "condition"]).size().unstack().to_string())
    print(rd.groupby(["label", "spn", "condition"])[["mean_log_rate", "sd_log_rate", "within_bout_corr",
                                                     "participation_ratio"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
