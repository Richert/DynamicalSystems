"""Lag between SPN activity and behaviour (model-free), per recording of the analysis
conditions, from the SPN data file (behaviour aligned with the fixed video offset).

Neural readouts (both, per cell): spikes smoothed with a Gaussian kernel (kernel_s) and the
denoised trace C / noise. Behaviour: log speed (log10(speed + 0.5 cm/s)).
  1. population cross-correlation: mean over cells (z-scored per cell) vs log speed, lags
     +-max_lag_s; lag > 0: behaviour follows the neural activity
  2. per-cell cross-correlation with log speed; cells whose peak |r| exceeds the 99th
     percentile of a circular-shift null (n_null shifts >= 30 s) are kept; peak lag per cell
  3. movement onsets (rest >= pre_s followed by a moving label >= post_s): onset-triggered
     averages of population activity and log speed; half-rise time of each

Outputs (OUT_DIR): lags_population.csv, lags_cells.csv.gz, lags_onsets.npz, lags.png

  conda run -n parker python analysis_spn_behavior_lags.py
"""
import os
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter1d

dataset = os.environ.get("DATASET", "/home/rgast/data/parker_data/dataset/clozapine_spn.h5")
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/spn_behavior_lags")
conditions = ["Control", "Amphetamine", "A + Clozapine high"]
fs = 20.0
kernel_s = 0.1
max_lag_s = 3.0
n_null, null_min_s, null_q = 20, 30.0, 99
pre_s, post_s, win_s = 1.0, 1.0, 3.0
seed = 1

def z(x, axis=-1):
    x = x - np.nanmean(x, axis, keepdims=True)
    return x / (np.nanstd(x, axis, keepdims=True) + 1e-12)

def xcorr(X, y, L):
    """Correlation of rows of X (z-scored) with y (z-scored) at lags -L..L
    (lag k > 0: y later than X)."""
    T = len(y)
    out = np.empty((X.shape[0], 2 * L + 1))
    for i, k in enumerate(range(-L, L + 1)):
        if k >= 0:
            out[:, i] = X[:, :T - k] @ y[k:] / (T - k)
        else:
            out[:, i] = X[:, -k:] @ y[:T + k] / (T + k)
    return out

def onsets(lab, rest, n_pre, n_post):
    """Frames where a rest run >= n_pre is followed by a moving run >= n_post."""
    moving = (lab >= 0) & (lab != rest)
    on = []
    st = np.flatnonzero(np.diff(np.r_[-2, lab]) != 0)
    en = np.r_[st[1:], len(lab)]
    for a, b, a2, b2 in zip(st[:-1], en[:-1], st[1:], en[1:]):
        if lab[a] == rest and b - a >= n_pre and moving[a2:a2 + n_post].all() and a2 + n_post <= len(lab):
            on.append(a2)
    return np.array(on, int)

def recording(r, rng):
    beh = r["behaviour"]
    names = list(beh.attrs["label_names"])
    lab = beh["label_clean"][:]
    sp = beh["speed"][:]
    ok = np.isfinite(sp)
    sp = np.log10(np.where(ok, sp, np.nanmedian(sp)) + 0.5)
    C, sn = r["C"][:], r["sn"][:]
    n, T = C.shape
    ptr, fr = r["event_ptr"][:], r["event_frames"][:]
    E = np.zeros((n, T), np.float32)
    for i in range(n):
        E[i, fr[ptr[i]:ptr[i + 1]]] = 1
    reads = {"spikes": gaussian_filter1d(E, kernel_s * fs, axis=1),
             "trace": (C / sn[:, None]).astype(np.float32)}
    L = int(max_lag_s * fs)
    y = z(sp)
    res = {}
    for key, X in reads.items():
        X = X[:, ok.nonzero()[0][0]:]                       # (speed nan only at the edges)
        Xz = z(X, 1)
        Xz[~np.isfinite(Xz)] = 0
        yy = y[ok.nonzero()[0][0]:][:X.shape[1]]
        pop = z(Xz.mean(0))
        pc = xcorr(pop[None], yy, L)[0]
        cc = xcorr(Xz, yy, L)
        null = []
        for _ in range(n_null):                             # circular shifts of the behaviour
            s = rng.integers(int(null_min_s * fs), len(yy) - int(null_min_s * fs))
            null.append(np.abs(xcorr(Xz, np.roll(yy, s), L)).max(1))
        thr = np.percentile(np.concatenate(null), null_q)
        pk = np.abs(cc).argmax(1)
        res[key] = dict(pop=pc, cell_lag=(pk - L) / fs, cell_r=cc[np.arange(n), pk],
                        cell_sig=np.abs(cc).max(1) > thr)
        # movement onsets
        on = onsets(lab, names.index("rest"), int(pre_s * fs), int(post_s * fs))
        W = int(win_s * fs)
        on = on[(on >= W) & (on + W < T)]
        full_pop = z(z(reads[key], 1).mean(0))
        res[key]["onset_pop"] = np.array([full_pop[o - W:o + W + 1] for o in on]) if len(on) else None
        res[key]["onset_speed"] = np.array([sp[o - W:o + W + 1] for o in on]) if len(on) else None
    return res

def half_rise(t, x, base=(-win_s, -1.0)):
    """Time where x first crosses half of its rise from the baseline window to its peak after."""
    b = x[(t >= base[0]) & (t < base[1])].mean()
    after = t >= base[1]
    p = x[after].max()
    k = np.flatnonzero(after & (x >= b + 0.5 * (p - b)))
    return t[k[0]] if len(k) else np.nan

def main():
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    pops, cells, ons = [], [], {}
    with h5py.File(dataset, "r") as h:
        for day in h:
            g = h[day]
            for rec in ("rec0", "rec1"):
                r = g[rec]
                cond = r.attrs["condition"]
                if cond not in conditions or not r["behaviour"].attrs.get("available", False):
                    continue
                res = recording(r, rng)
                base = dict(day=day, rec=rec, condition=cond, mouse=g.attrs["mouse"], spn=g.attrs["spn"])
                for key, v in res.items():
                    L = (len(v["pop"]) - 1) // 2
                    pops.append(dict(base, readout=key, peak_lag=(np.argmax(v["pop"]) - L) / fs,
                                     peak_r=v["pop"].max(), **{f"r{k}": v["pop"][k] for k in range(len(v["pop"]))}))
                    cells.append(pd.DataFrame(dict(base, readout=key, lag=v["cell_lag"], r=v["cell_r"],
                                                   sig=v["cell_sig"])))
                    if v["onset_pop"] is not None:
                        ons.setdefault((key, cond, g.attrs["spn"]), []).append(
                            (v["onset_pop"].mean(0), v["onset_speed"].mean(0), len(v["onset_pop"])))
            print(day, flush=True)
    pop = pd.DataFrame(pops)
    cel = pd.concat(cells, ignore_index=True)
    pop.to_csv(f"{out_dir}/lags_population.csv", index=False)
    cel.to_csv(f"{out_dir}/lags_cells.csv.gz", index=False)
    t = np.arange(-int(win_s * fs), int(win_s * fs) + 1) / fs
    summ = []
    arr = {}
    for (key, cond, spn), v in ons.items():
        P = np.array([a for a, _, _ in v]); S = np.array([b for _, b, _ in v])
        arr[f"{key}|{cond}|{spn}|pop"], arr[f"{key}|{cond}|{spn}|speed"] = P, S
        rise_p = [half_rise(t, p) for p in P]
        rise_s = [half_rise(t, s) for s in S]
        summ.append(dict(readout=key, condition=cond, spn=spn, recordings=len(v),
                         onsets=sum(c for _, _, c in v), rise_pop=np.nanmedian(rise_p),
                         rise_speed=np.nanmedian(rise_s),
                         lead=np.nanmedian(np.array(rise_s) - np.array(rise_p))))
    np.savez(f"{out_dir}/lags_onsets.npz", t=t, **arr)
    pd.set_option("display.width", 200)
    print(pop.groupby(["readout", "spn", "condition"])[["peak_lag", "peak_r"]].median().round(3).to_string())
    s = cel[cel.sig]
    print(cel.groupby(["readout", "spn"]).sig.mean().round(3).to_string())
    print(s.groupby(["readout", "spn", "condition"]).agg(n=("lag", "size"), med_lag=("lag", "median"),
                                                         pos_r=("r", lambda x: np.mean(x > 0))).round(3).to_string())
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    figure(pop, cel, arr, t)

def figure(pop, cel, arr, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = dict(zip(conditions, ["#2a78d6", "#eda100", "#e34948"]))
    L = int(max_lag_s * fs)
    lags = np.arange(-L, L + 1) / fs
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5), layout="constrained")
    for j, (key, spn) in enumerate([(k, s) for k in ("spikes", "trace") for s in ("D1", "D2")]):
        ax = axes[0, j]
        for c in conditions:
            d = pop[(pop.readout == key) & (pop.spn == spn) & (pop.condition == c)]
            R = d[[f"r{k}" for k in range(2 * L + 1)]].values
            if len(R):
                m, se = R.mean(0), R.std(0) / np.sqrt(len(R))
                ax.plot(lags, m, color=cols[c], label=f"{c} ({len(R)})")
                ax.fill_between(lags, m - se, m + se, color=cols[c], alpha=0.2, lw=0)
        ax.axvline(0, color="0.5", lw=0.7, ls=":")
        ax.set(title=f"{spn}, {key}: population vs log speed", xlabel="lag (s; > 0: behaviour follows)",
               ylabel="correlation")
        ax.legend(fontsize=7, frameon=False)
        ax = axes[1, j]
        s = cel[(cel.readout == key) & (cel.spn == spn) & cel.sig]
        bins = np.arange(-max_lag_s, max_lag_s + 0.1, 0.1) - 0.05
        for c in conditions:
            x = s[s.condition == c]
            ax.hist(x.lag[x.r > 0], bins=bins, histtype="step", density=True, color=cols[c],
                    label=f"{c} ({len(x)}, {np.mean(x.r > 0):.0%} r > 0)")
        ax.axvline(0, color="0.5", lw=0.7, ls=":")
        ax.set(title=f"{spn}, {key}: peak lag of speed-tuned cells (r > 0)", xlabel="lag (s)")
        ax.legend(fontsize=7, frameon=False)
        ax = axes[2, j]
        for c in conditions:
            kp, ks = f"{key}|{c}|{spn}|pop", f"{key}|{c}|{spn}|speed"
            if kp in arr:
                ax.plot(t, z(arr[kp].mean(0)), color=cols[c], label=f"{c}: SPN")
                ax.plot(t, z(arr[ks].mean(0)), color=cols[c], ls="--", lw=1, label=f"{c}: speed")
        ax.axvline(0, color="0.5", lw=0.7, ls=":")
        ax.set(title=f"{spn}, {key}: movement onset (rest -> moving)", xlabel="time from onset (s)",
               ylabel="z (mean over recordings)")
        ax.legend(fontsize=6.5, frameon=False, ncol=2)
    for a in axes.flat:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.savefig(f"{out_dir}/lags.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
