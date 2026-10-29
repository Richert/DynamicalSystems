"""Behaviour annotations (MARS/Bento) per drug condition: occupancy, bout durations and
inter-bout intervals, combining the automated classifiers ("channels") per behaviour.

The Bento .annot files hold, per channel (a classifier trained by one lab, e.g.
"AdaptableSnail"), bouts (start/stop frame, 1-based, inclusive) per behaviour. A
behaviour section only appears if that classifier detected >= 1 bout, so the set of
classifiers is taken from the .probs.csv headers (channel__behaviour columns) and a
missing section means "no bouts". Per behaviour, the classifiers are combined frame
by frame:
  or        frame labelled if ANY of the behaviour's classifiers labels it
  majority  frame labelled if MORE THAN HALF of the behaviour's classifiers label it
            (single-classifier behaviours: identical to "or")
and bouts are re-extracted from the combined frames.

By default only the imaging period of each video is analysed (offsets from
align_pose_to_imaging.py), so behaviour matches the neural data; set window =
"video" for the whole video (which also contains the 10-min post-injection wait).

Per session: occupancy, bouts/min, median bout duration and median inter-bout
interval; conditions compared with value ~ C(condition) + (1|mouse) (all mice
pooled, BH-FDR per behaviour x measure). Figures: one per combination method.

  /home/rgast/conda/envs/arhmm/bin/python analysis_behavior_annotations.py
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd

pose_root = "/home/rgast/data/parker_data/pose_data/Clozapine"            # local copy
offsets_csv = "/home/rgast/data/parker_data/pose_alignment/pose_imaging_offsets.csv"
out_dir = "/home/rgast/data/parker_data/behavior_annotations"
drug = "Clozapine"
fs = 20.0
window = "imaging"                 # "imaging" | "video"
methods = ["or"]                   # "or" | "majority" (majority: see module docstring)
# analysed behaviours: groups of classifier behaviours, combined by OR. "huddle" (a social
# classifier applied to a single mouse) tracks "rest" closely and is merged into it
behaviour_groups = {"rest": ["rest", "huddle"], "freeze": ["freeze"], "selfgroom": ["selfgroom"]}
# smoothing of the combined frames (classifier outputs flicker): first merge bouts
# separated by gaps < max_gap_s, then drop bouts shorter than min_bout_s
max_gap_s = 0.5
min_bout_s = 0.5
conditions = ["Control", drug, "Amphetamine", f"A + {drug}"]
cond_of = {("Vehicle", "Control"): "Control", ("Vehicle", "Amph"): "Amphetamine",
           ("HighDose", "Control"): drug, ("HighDose", "Amph"): f"A + {drug}"}
contrasts = [(drug, "Control"), ("Amphetamine", "Control"), (f"A + {drug}", "Control"),
             (f"A + {drug}", "Amphetamine"), (f"A + {drug}", drug)]
measures = ["occupancy", "bouts_per_min", "bout_med_s", "ibi_med_s"]
min_bouts_session = 5          # per-session median bout / IBI only with >= this many bouts
min_bouts_pooled = 20          # distribution curves only with >= this many pooled bouts
any_label = "any label"        # union of all behaviours: how much time is labelled at all

# figure style: validated 4-slot categorical set + secondary encoding (line style)
colors = dict(zip(conditions, ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]))
styles = dict(zip(conditions, ["-", "--", ":", "-."]))
ink, ink2, grid_c, mouse_c = "#1f1f1e", "#5f5e5a", "#e6e5e0", "#c9c8c2"

# ---------------------------------------------------------------------------
# parsing and combining
# ---------------------------------------------------------------------------

def parse_annot(f):
    """(n_frames, {(channel, behaviour): [(start, stop), ...]}) with 0-based,
    inclusive frame indices."""
    bouts, ch, beh, n = {}, None, None, None
    for line in open(f):
        line = line.rstrip("\n")
        if line.startswith("Annotation stop frame"):
            n = int(float(line.split(":")[1]))
        m = re.match(r"^(\w+)-{5,}$", line)
        if m:
            ch, beh = m.group(1), None
            continue
        if line.startswith(">"):
            beh = line[1:].strip()
            bouts.setdefault((ch, beh), [])
            continue
        m = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)", line)
        if m and ch and beh:
            bouts[(ch, beh)].append((int(m.group(1)) - 1, int(m.group(2)) - 1))
    return n, bouts

def classifier_set(files):
    """{behaviour: [channels]} from the .probs.csv headers (all classifiers that ran)."""
    pairs = set()
    for f in files:
        p = f.replace(".annot", ".probs.csv")
        if os.path.isfile(p):
            pairs |= {tuple(c.split("__")) for c in open(p).readline().strip().split(",")[1:]}
        else:                                           # fall back to the annotated pairs
            pairs |= set(parse_annot(f)[1])
    out = {}
    for ch, beh in sorted(pairs):
        out.setdefault(beh, []).append(ch)
    return out

def combined_masks(n, bouts, classifiers, method):
    """{behaviour: boolean frame mask} combining the behaviour's classifiers."""
    masks = {}
    for beh, chans in classifiers.items():
        votes = np.zeros(n, int)
        for ch in chans:
            m = np.zeros(n, bool)
            for s, e in bouts.get((ch, beh), []):
                m[s:e + 1] = True
            votes += m
        masks[beh] = votes > 0 if method == "or" else votes > len(chans) / 2
    return masks

def smooth(mask):
    """Fill gaps shorter than max_gap_s between bouts, then remove bouts shorter
    than min_bout_s."""
    m = mask.copy()
    d = np.diff(np.r_[0, m.astype(int), 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    for a, b in zip(en[:-1], st[1:]):                      # gaps between bouts
        if (b - a) / fs < max_gap_s:
            m[a:b] = True
    d = np.diff(np.r_[0, m.astype(int), 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    for a, b in zip(st, en):
        if (b - a) / fs < min_bout_s:
            m[a:b] = False
    return m

def bouts_and_gaps(mask):
    """Bout durations and inter-bout intervals (s) of a boolean mask. Bouts touching
    the start or end of the analysis window are cut off (true duration unknown) and
    are left out of the durations (they still count towards occupancy)."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    full = (st > 0) & (en < len(mask))
    return (en - st)[full] / fs, (st[1:] - en[:-1]) / fs

# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def condition_stats(df, q):
    """value ~ C(condition) + (1|mouse) per behaviour; Wald contrasts; BH-FDR."""
    import statsmodels.api as sm
    from patsy import dmatrices
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    import analysis_arhmm_spn_states as A
    out = []
    for beh, d in df.dropna(subset=[q]).groupby("behavior"):
        d = d.copy()
        d["condition"] = pd.Categorical(d.condition, categories=[c for c in conditions
                                                                 if c in set(d.condition)])
        if d.condition.nunique() < 2 or d[q].std() == 0:
            continue
        y, X = dmatrices(f"{q} ~ C(condition)", d, return_type="dataframe")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m = A._fit_mixedlm(y, X, d["mouse"].values)
                ok = A._valid_fit(m, "mixedlm", d.mouse.nunique())[0]
            except Exception:
                ok = False
            if not ok:
                m = sm.OLS(y, X).fit(cov_type="cluster",
                                     cov_kwds={"groups": d.mouse.astype("category").cat.codes})
        p, cov = m.params, m.cov_params()
        cats = list(d.condition.cat.categories)
        def vec(c):
            v = pd.Series(0.0, index=p.index)
            v["Intercept"] = 1.0
            if c != cats[0]:
                v[f"C(condition)[T.{c}]"] = 1.0
            return v
        rows = []
        for a, b in contrasts:
            if a in cats and b in cats:
                v = vec(a) - vec(b)
                est, se = float(v @ p), float(np.sqrt(v @ cov.loc[p.index, p.index] @ v))
                rows.append(dict(behavior=beh, measure=q, contrast=f"{a} - {b}", diff=est,
                                 se=se, p=2 * norm.sf(abs(est / se))))
        if rows:
            r = pd.DataFrame(rows)
            r["q_fdr"] = multipletests(r.p, method="fdr_bh")[1]
            out.append(r)
    return pd.concat(out) if out else pd.DataFrame()

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(f"{pose_root}/*/*/*.annot"))
    classifiers = classifier_set(files)
    group_cls = {g: sum((classifiers.get(b, []) for b in members), [])
                 for g, members in behaviour_groups.items()}
    print("classifiers per analysed behaviour:", {g: len(c) for g, c in group_cls.items()})
    offsets = pd.read_csv(offsets_csv).set_index("session") if window == "imaging" else None

    rows, pooled = [], {}
    for f in files:
        dose, part = f.split("/")[-3:-1]
        cond = cond_of.get((dose, part))
        if cond is None:
            continue
        base = os.path.basename(f).split("_pose_top")[0]
        mouse = re.search(r"_(m\d+|f\d+)_", base).group(1)
        n, bouts = parse_annot(f)
        if window == "imaging":
            o = offsets.loc[f"{drug.lower()}/{dose}/{base}"]
            sl = slice(int(o.offset_frames), int(o.offset_frames) + int(o.old_frames_5hz) * 4)
        else:
            sl = slice(0, n)
        minutes = (sl.stop - sl.start) / fs / 60
        for method in methods:
            raw = combined_masks(n, bouts, classifiers, method)
            masks = {g: smooth(np.any([raw[b] for b in members if b in raw], axis=0))
                     for g, members in behaviour_groups.items()}
            masks[any_label] = np.any(list(masks.values()), axis=0)
            for beh, mask in masks.items():
                m = mask[sl]
                dur, gaps = bouts_and_gaps(m)
                enough = len(dur) >= min_bouts_session
                rows.append(dict(session=base, mouse=mouse, condition=cond, method=method,
                                 behavior=beh,
                                 n_classifiers=sum(map(len, group_cls.values()))
                                 if beh == any_label else len(group_cls[beh]),
                                 occupancy=m.mean(), n_bouts=len(dur),
                                 bouts_per_min=len(dur) / minutes,
                                 bout_med_s=np.median(dur) if enough else np.nan,
                                 ibi_med_s=np.median(gaps) if enough else np.nan))
                key = (method, beh, cond)
                pooled.setdefault(key, [[], []])
                pooled[key][0].append(dur)
                pooled[key][1].append(gaps)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/behavior_per_session_{window}.csv", index=False)
    long = [dict(method=k[0], behavior=k[1], condition=k[2], kind=kind, value_s=v)
            for k, (durs, gaps) in pooled.items()
            for kind, arrs in (("bout", durs), ("ibi", gaps)) for a in arrs for v in a]
    pd.DataFrame(long).to_csv(f"{out_dir}/bouts_and_intervals_{window}.csv", index=False)

    stats = []
    for method in methods:
        for q in measures:
            s = condition_stats(df[df.method == method], q)
            if len(s):
                stats.append(s.assign(method=method))
    stats = pd.concat(stats)
    stats.to_csv(f"{out_dir}/behavior_contrasts_{window}.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nmean occupancy per condition (%):")
    print((df.groupby(["method", "behavior", "condition"]).occupancy.mean().unstack()[conditions]
           * 100).round(2).to_string())
    print("\nsignificant contrasts (q < 0.05):")
    sig = stats[stats.q_fdr < 0.05]
    print(sig[["method", "behavior", "measure", "contrast", "diff", "q_fdr"]].round(4)
          .to_string(index=False) if len(sig) else "  none")
    for method in methods:
        make_figure(df[df.method == method], pd.DataFrame(long), stats, method, group_cls)

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def style_axes(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink2)
    ax.tick_params(colors=ink2, labelsize=7.5)
    if ygrid:
        ax.yaxis.grid(True, color=grid_c, lw=0.8)
    ax.set_axisbelow(True)

def stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*"

def occupancy_panel(ax, d, st):
    pm = (d.groupby(["mouse", "condition"]).occupancy.mean().unstack()
          .reindex(columns=conditions) * 100)
    x = np.arange(len(conditions))
    for _, row in pm.iterrows():
        ok = row.notna().values
        ax.plot(x[ok], row.values[ok], color=mouse_c, lw=0.7, zorder=1)
        ax.scatter(x[ok], row.values[ok], color=mouse_c, s=8, zorder=1)
    for i, c in enumerate(conditions):
        ax.errorbar(i, pm[c].mean(), yerr=pm[c].sem(), fmt="o", ms=6.5, color=colors[c],
                    mec=ink, mew=0.6, ecolor=ink, elinewidth=1.1, capsize=3, zorder=3)
    lo, hi = np.nanmin(pm.values), np.nanmax(pm.values)
    rng = max(hi - lo, 1e-6)
    level, step = hi + rng * 0.06, rng * 0.1
    for t in st.sort_values("contrast").itertuples():
        a, b = t.contrast.split(" - ")
        i, j = conditions.index(a), conditions.index(b)
        ax.plot([i, i, j, j], [level - step * 0.3, level, level, level - step * 0.3],
                color=ink2, lw=0.8)
        ax.text((i + j) / 2, level, stars(t.q_fdr), ha="center", va="bottom", fontsize=7.5,
                color=ink)
        level += step
    ax.set_ylim(max(lo - rng * 0.08, 0 if lo >= 0 else lo - rng), level + step * 0.2)
    ax.set_yticks([t for t in ax.get_yticks() if t <= hi + rng * 0.02])
    ax.set_xticks(x, ["Ctrl", "Clz", "Amph", "A+Clz"], fontsize=7.5, color=ink)
    ax.set_ylabel("occupancy (% of time)", fontsize=8, color=ink)
    style_axes(ax)

def ecdf_panel(ax, vals, kind):
    for c in conditions:
        v = np.sort(vals.get(c, np.array([])))
        if len(v) < min_bouts_pooled:
            continue
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=colors[c],
                ls=styles[c], lw=1.6, label=f"{c}: median {np.median(v):.2g} s (n={len(v)})")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{'bout duration' if kind == 'bout' else 'inter-bout interval'} (s)",
                  fontsize=8, color=ink)
    ax.set_ylabel("cumulative fraction", fontsize=8, color=ink)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6.3, frameon=False, loc="lower right", labelcolor=ink, handlelength=2.2)
    else:
        ax.text(0.5, 0.5, f"< {min_bouts_pooled} bouts per condition", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=ink2)
    style_axes(ax)

def make_figure(df, long, stats, method, classifiers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    behs = [b for b in df.groupby("behavior").occupancy.mean().sort_values(ascending=False).index
            if b != any_label]
    behs = [any_label] + behs
    blocks = 1 if len(behs) <= 5 else 2
    half = int(np.ceil(len(behs) / blocks))
    fig, axes = plt.subplots(half, 3 * blocks, figsize=(11.5 * blocks, 3.2 * half + 1.4),
                             layout="constrained", squeeze=False,
                             gridspec_kw=dict(width_ratios=[0.8, 1, 1] * blocks))
    lm = long[long.method == method]
    st_m = stats[(stats.method == method) & (stats.measure == "occupancy") & (stats.q_fdr < 0.05)]
    for k, beh in enumerate(behs):
        r, c0 = k % half, 3 * (k // half)
        d = df[df.behavior == beh]
        occupancy_panel(axes[r, c0], d, st_m[st_m.behavior == beh])
        if beh == any_label:
            title = "any of the three behaviours: labelled share of time"
        else:
            n_cls = len(classifiers[beh])
            rule = "" if n_cls == 1 else (" (any)" if method == "or" else f" (> {n_cls / 2:g} of {n_cls})")
            title = f"{beh}: {n_cls} classifier{'s' if n_cls > 1 else ''}{rule}"
        axes[r, c0].set_title(title, fontsize=10, color=ink, loc="left")
        for j, kind in enumerate(["bout", "ibi"], start=1):
            sub = lm[(lm.behavior == beh) & (lm.kind == kind)]
            ecdf_panel(axes[r, c0 + j], {c: g.value_s.values for c, g in sub.groupby("condition")},
                       kind)
    for k in range(len(behs), blocks * half):
        for j in range(3):
            axes[k % half, 3 * (k // half) + j].axis("off")
    win = "imaging periods" if window == "imaging" else "whole videos"
    combine = "OR (any classifier)" if method == "or" else "majority vote of classifiers"
    fig.suptitle(f"Behaviour per drug condition ({win})\n"
                 f"MARS/Bento classifiers combined by {combine}; gaps < {max_gap_s:g} s merged, "
                 f"bouts < {min_bout_s:g} s dropped\n"
                 f"left: occupancy (grey: mice, coloured: mean ± SEM; brackets: mixed model "
                 f"~ condition + (1|mouse), BH-FDR q < 0.05)\n"
                 f"middle / right: bout durations (bouts cut by the window edges excluded) and "
                 f"inter-bout intervals, pooled over sessions (curves need >= {min_bouts_pooled} bouts)\n"
                 f"Ctrl = Control, Clz = {drug}, Amph = Amphetamine, A+Clz = A + {drug}",
                 fontsize=10.5, color=ink)
    out = f"{out_dir}/behavior_summary_{method}_{window}.png"
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
