"""SPN activity vs one behaviour feature (default: speed) per drug condition: population tuning
curves and the per-cell lag of the maximal correlation (spikes; from the SPN data file).

Per recording: spikes smoothed with a Gaussian kernel (kernel_s), feature in its analysis
transform (speed / head speed: log10(x + 0.5 cm/s), turning: arcsinh(x / 5 deg/s), others raw).
  per cell   cross-correlation with the feature over +-max_lag_s (lag > 0: behaviour follows the
             SPN); peak = largest |r|; significant if it exceeds the null_q percentile of the
             same statistic with the behaviour circularly shifted (n_null shifts >= 30 s)
  tuning     population event rate (mean over cells, Hz) in feature bins (common bins across
             recordings), with the neural activity shifted by the recording's population peak lag
             (bins with >= min_bin_s of data)
Conditions: Control, Amphetamine, <drug> high, A + <drug> high (DRUG, default Clozapine).

Outputs (OUT_DIR): tuning_<feature>.csv, lags_<feature>.csv.gz, figure tuning_<feature>.png/pdf.

  conda run -n parker python analysis_spn_feature_tuning.py [feature]
"""
import os
import sys
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter1d

drug = os.environ.get("DRUG", "Clozapine")
dataset = os.environ.get("DATASET", f"/home/rgast/data/parker_data/dataset/{drug.lower()}_spn.h5")
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/spn_feature_tuning")
slide_dir = os.environ.get("SLIDE_DIR", "/home/rgast/OneDrive/presentations/parker/images")
feature = sys.argv[1] if len(sys.argv) > 1 else "speed"
conditions = ["Control", "Amphetamine", f"{drug} high", f"A + {drug} high"]
fs = 20.0
kernel_s = 0.1
max_lag_s = 2.0
n_null, null_min_s, null_q = 20, 30.0, 99
min_bin_s = 5.0
seed = 1
TRANSFORM = {"speed": lambda x: np.log10(x + 0.5), "head_speed": lambda x: np.log10(x + 0.5),
             "turn": lambda x: np.arcsinh(x / 5.0), "head_turn": lambda x: np.arcsinh(x / 5.0)}
BINS = {"speed": np.log10(np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 18, 27, 40]) + 0.5),
        "head_speed": np.log10(np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 18, 27, 40]) + 0.5)}
LABEL = {"speed": "speed (cm/s)", "head_speed": "head speed (cm/s)", "turn": "turning (asinh, deg/s)",
         "head_angle": "head angle (deg)", "bend": "body bend (deg)", "wall_dist": "wall distance (cm)",
         "wall_ahead": "wall ahead (cm)", "body_len": "body length (cm)"}

def z(x, axis=-1):
    x = x - np.nanmean(x, axis, keepdims=True)
    return x / (np.nanstd(x, axis, keepdims=True) + 1e-12)

def xcorr(X, y, L):
    T = len(y)
    out = np.empty((X.shape[0], 2 * L + 1))
    for i, k in enumerate(range(-L, L + 1)):
        out[:, i] = (X[:, :T - k] @ y[k:] / (T - k)) if k >= 0 else (X[:, -k:] @ y[:T + k] / (T + k))
    return out

def recording(r, rng, bins):
    x = r["behaviour"][feature][:]
    ok = np.isfinite(x)
    a, b = np.flatnonzero(ok)[[0, -1]]
    x = TRANSFORM.get(feature, lambda v: v)(np.where(ok, x, np.nanmedian(x)))[a:b + 1]
    ptr, fr = r["event_ptr"][:], r["event_frames"][:]
    n, T = len(ptr) - 1, r["C"].shape[1]
    E = np.zeros((n, T), np.float32)
    for i in range(n):
        E[i, fr[ptr[i]:ptr[i + 1]]] = 1
    E = E[:, a:b + 1]
    S = gaussian_filter1d(E, kernel_s * fs, axis=1)
    L = int(max_lag_s * fs)
    Xz, y = z(S, 1), z(x)
    Xz[~np.isfinite(Xz)] = 0
    cc = xcorr(Xz, y, L)
    null = [np.abs(xcorr(Xz, np.roll(y, rng.integers(int(null_min_s * fs), len(y) - int(null_min_s * fs))),
                         L)).max(1) for _ in range(n_null)]
    thr = np.percentile(np.concatenate(null), null_q)
    pk = np.abs(cc).argmax(1)
    pop = xcorr(z(S.mean(0))[None], y, L)[0]
    lag = int(np.argmax(np.abs(pop))) - L                  # population peak lag (frames)
    # tuning: population rate at frame t vs feature at t + lag
    rate = E.mean(0) * fs
    rate = gaussian_filter1d(rate, kernel_s * fs)
    if lag >= 0:
        rr, xx = rate[:len(rate) - lag], x[lag:]
    else:
        rr, xx = rate[-lag:], x[:len(x) + lag]
    idx = np.digitize(xx, bins) - 1
    tune = [rr[idx == k].mean() if (idx == k).sum() / fs >= min_bin_s else np.nan
            for k in range(len(bins) - 1)]
    return dict(cell_lag=(pk - L) / fs, cell_r=cc[np.arange(n), pk], cell_sig=np.abs(cc).max(1) > thr,
                pop_lag=lag / fs, pop_r=pop[lag + L], tuning=np.array(tune))

def main():
    os.makedirs(out_dir, exist_ok=True)
    sfx = "" if drug == "Clozapine" else "_" + drug.lower()
    if os.environ.get("FIGURE_ONLY"):
        figure(pd.read_csv(f"{out_dir}/tuning_{feature}{sfx}.csv"),
               pd.read_csv(f"{out_dir}/lags_{feature}{sfx}.csv.gz"),
               BINS[feature] if feature in BINS else np.load(f"{out_dir}/bins_{feature}{sfx}.npy"))
        return
    rng = np.random.default_rng(seed)
    with h5py.File(dataset, "r") as h:
        if feature in BINS:
            bins = BINS[feature]
        else:                                                # common quantile bins (Control)
            v = np.concatenate([TRANSFORM.get(feature, lambda q: q)(h[d][r]["behaviour"][feature][:])
                                for d in h for r in ("rec0", "rec1")
                                if h[d][r].attrs["condition"] == "Control"
                                and h[d][r]["behaviour"].attrs.get("available", False)])
            bins = np.nanpercentile(v, np.linspace(1, 99, 11))
        tun, cells = [], []
        for day in h:
            for rec in ("rec0", "rec1"):
                r = h[day][rec]
                if r.attrs["condition"] not in conditions or not r["behaviour"].attrs.get("available", False):
                    continue
                res = recording(r, rng, bins)
                base = dict(day=day, rec=rec, condition=r.attrs["condition"], mouse=h[day].attrs["mouse"],
                            spn=h[day].attrs["spn"])
                tun.append(dict(base, pop_lag=res["pop_lag"], pop_r=res["pop_r"],
                                **{f"b{k}": v for k, v in enumerate(res["tuning"])}))
                cells.append(pd.DataFrame(dict(base, lag=res["cell_lag"], r=res["cell_r"], sig=res["cell_sig"])))
            print(day, flush=True)
    tu, ce = pd.DataFrame(tun), pd.concat(cells, ignore_index=True)
    np.save(f"{out_dir}/bins_{feature}{'' if drug == 'Clozapine' else '_' + drug.lower()}.npy", bins)
    tu.to_csv(f"{out_dir}/tuning_{feature}{'' if drug == 'Clozapine' else '_' + drug.lower()}.csv", index=False)
    ce.to_csv(f"{out_dir}/lags_{feature}{'' if drug == 'Clozapine' else '_' + drug.lower()}.csv.gz", index=False)
    pd.set_option("display.width", 200)
    print(tu.groupby(["spn", "condition"])[["pop_lag", "pop_r"]].median().round(3).to_string())
    s = ce[ce.sig]
    print(ce.groupby(["spn", "condition"]).sig.mean().round(3).unstack().to_string())
    print(s.groupby(["spn", "condition"]).agg(n=("lag", "size"), med_lag=("lag", "median"),
                                             pos=("r", lambda q: np.mean(q > 0))).round(3).to_string())
    figure(tu, ce, bins)

def figure(tu, ce, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    SPN_COL = {"D1": "#1f8a70", "D2": "#8a4fb5"}
    COND_COL = dict(zip(conditions, ["#2a78d6", "#eda100", "#8a4fb5", "#e34948"]))
    short = dict(zip(conditions, ["Control", "Amph.", f"{drug[:4]}. high", f"Amph. + {drug[:4]}. high"]))
    plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.7,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.7), sharey="row")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.14, wspace=0.12, hspace=0.75)
    centres = (bins[:-1] + bins[1:]) / 2
    nb = len(centres)
    for j, c in enumerate(conditions):
        ax = axes[0, j]
        for spn in ("D1", "D2"):
            d = tu[(tu.condition == c) & (tu.spn == spn)]
            T = d[[f"b{k}" for k in range(nb)]].values
            for row in T:
                ax.plot(centres, row, color=SPN_COL[spn], lw=0.5, alpha=0.3)
            m = np.nanmean(T, 0)
            se = np.nanstd(T, 0) / np.sqrt(np.sum(np.isfinite(T), 0))
            ok = np.sum(np.isfinite(T), 0) >= 3
            ax.plot(centres[ok], m[ok], color=SPN_COL[spn], lw=1.8, label=f"{spn} ({len(d)})")
            ax.fill_between(centres[ok], (m - se)[ok], (m + se)[ok], color=SPN_COL[spn], alpha=0.2, lw=0)
        if feature in ("speed", "head_speed"):
            t = [0, 1, 3, 10, 30]
            ax.set_xticks(np.log10(np.array(t) + 0.5), [str(v) for v in t])
        ax.set_xlabel(LABEL.get(feature, feature))
        ax.set_title(short[c], loc="left", color=INK, fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("population event rate (Hz)")
            ax.legend(frameon=False, loc="upper left", title="mice", title_fontsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        # lags of significantly correlated cells
        ax = axes[1, j]
        bl = np.arange(-max_lag_s, max_lag_s + 0.1, 0.1) - 0.05
        for spn in ("D1", "D2"):
            d = ce[(ce.condition == c) & (ce.spn == spn)]
            s = d[d.sig]
            if not len(s):
                continue
            h, _ = np.histogram(s.lag, bins=bl)
            ax.stairs(h / len(s), bl, color=SPN_COL[spn], lw=1.3)
            med = s.lag.median()
            ax.axvline(med, color=SPN_COL[spn], lw=0.8, ls=(0, (3, 2)))
            ax.text(0.0, 1.03 + 0.12 * (spn == "D1"), f"{spn}: {d.sig.mean():.0%} · {np.mean(s.r > 0):.0%} · "
                    f"{med:+.2f} s", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=5.6, color=SPN_COL[spn])
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xlim(-max_lag_s, max_lag_s)
        ax.set_xlabel("lag of max. |r| (s)")
        if j == 0:
            ax.set_ylabel("share of correlated cells")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.text(0.09, 0.955, f"SPN activity vs {LABEL.get(feature, feature).split(' (')[0]} ({drug} cohort; "
             f"lag > 0: behaviour follows the SPNs)", fontsize=8.5, color=INK)
    fig.text(0.09, 0.015, "lag panels: share of cells correlated (vs. shuffled behaviour) · share of those with "
             "r > 0 · median lag (dashed)", fontsize=6.3, color=INK2)
    tagname = f"spn_tuning_{feature}" + ("" if drug == "Clozapine" else f"_{drug.lower()}")
    for d in (out_dir, slide_dir):
        os.makedirs(d, exist_ok=True)
        fig.savefig(f"{d}/{tagname}.pdf")
        fig.savefig(f"{d}/{tagname}.png", dpi=200)
    plt.close(fig)
    print(f"saved {tagname}.pdf/.png")


if __name__ == "__main__":
    main()
