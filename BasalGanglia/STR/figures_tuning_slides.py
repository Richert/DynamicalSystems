"""Slide figures of the per-cell feature tuning (analysis_spn_feature_tuning_cells.py, equal-occupancy
version: the same time per feature bin in every condition), both cohorts, built up per condition.

  tuning_matrix   share of tuned cells (dot size) and median tuning strength eta^2 of the tuned
                  cells (colour) per feature x condition, D1 / D2, clozapine and haloperidol cohort
  tuning_curves   population tuning curves (each tuned cell's curve / its mean over bins; mean per
                  mouse, mean +- SEM over mice with >= min_cells tuned cells) for speed and turning

Frames <name>_s1 ... _s4: Control, + Amphetamine, + drug alone, + Amphetamine + drug. The tuning-curve
panels have an inset with the share of tuned cells (the cells the curves average over).

  conda run -n parker python figures_tuning_slides.py
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

res_dir = os.environ.get("RES_DIR", "/home/rgast/data/parker_data/spn_feature_tuning_cells")
out_dir = os.environ.get("SLIDE_DIR", "/home/rgast/OneDrive/presentations/parker/images")
drugs = ["Clozapine", "Haloperidol"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COND_COL = ["#2a78d6", "#eda100", "#8a4fb5", "#e34948"]
COND_SHORT = ["Control", "Amph.", "drug", "Amph. + drug"]
GROUPS = [("movement", ["speed", "head_speed", "turn", "head_turn"]),
          ("posture", ["head_angle", "bend", "body_len", "ear_width", "hip_width"]),
          ("location", ["wall_dist", "wall_ahead"])]
NAMES = {"speed": "speed", "head_speed": "head speed", "turn": "turning", "head_turn": "head turning",
         "head_angle": "head angle", "bend": "bend", "body_len": "body length", "ear_width": "ear width",
         "hip_width": "hip width", "wall_dist": "wall distance", "wall_ahead": "wall ahead"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#cfe0f6", "#6da7ec", "#1c5cab", "#0b2f5f"])
min_cells = 3
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.7,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

def conditions(drug):
    return ["Control", "Amphetamine", f"{drug} high", f"A + {drug} high"]

def load(drug):
    return pd.read_csv(f"{res_dir}/cells_{drug.lower()}_eq.csv.gz")

def save_steps(fig, steps, name):
    managed = set().union(*steps)
    for k, vis in enumerate(steps, 1):
        for a in managed:
            a.set_visible(a in vis)
        fig.savefig(f"{out_dir}/{name}_s{k}.pdf")
        fig.savefig(f"{out_dir}/{name}_s{k}.png", dpi=150)
    for a in managed:
        a.set_visible(True)
    print(f"saved {out_dir}/{name}_s1..{len(steps)}")

def tuning_matrix():
    feats = [f for _, fs in GROUPS for f in fs]
    ypos, y = {}, 0.0
    for _, fs in GROUPS:
        for f in fs:
            ypos[f] = y
            y += 1
        y += 0.5
    data = {d: load(d) for d in drugs}
    vmax = max(c[c.tuned].groupby(["spn", "condition", "feature"]).eta2.median().max() for c in data.values())
    norm = Normalize(0, vmax)
    fig, axes = plt.subplots(1, 2, figsize=(6.15, 2.85), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.87, top=0.8, bottom=0.03, wspace=0.08)
    by_cond = [set() for _ in range(4)]
    for ax, drug in zip(axes, drugs):
        c = data[drug]
        conds = conditions(drug)
        for g, spn in enumerate(("D1", "D2")):
            for k, cond in enumerate(conds):
                x = g * 4.8 + k
                d = c[(c.spn == spn) & (c.condition == cond)]
                share = d.groupby("feature").tuned.mean()
                eta = d[d.tuned].groupby("feature").eta2.median()
                xs = [x] * len(feats)
                ys = [ypos[f] for f in feats]
                sz = [260 * share.get(f, 0) for f in feats]
                col = [SEQ(norm(eta.get(f, 0))) for f in feats]
                by_cond[k].add(ax.scatter(xs, ys, s=sz, c=col, edgecolor="white", lw=0.4, zorder=3))
                lab = ax.text(x, -1.1, COND_SHORT[k], rotation=50, ha="left", va="bottom", fontsize=6.3,
                              color=COND_COL[k], rotation_mode="anchor")
                by_cond[k].add(lab)
            ax.text(g * 4.8 + 1.5, -3.3, f"{spn} SPNs", ha="center", fontsize=7.5, color=INK)
        ax.set_title(f"{drug} cohort", loc="center", fontsize=8.5, pad=26, fontweight="bold")
        ax.set_xlim(-0.7, 8.5)
        ax.set_ylim(y - 0.2, -3.6)
        ax.set_xticks([])
        ax.set_yticks([ypos[f] for f in feats], [NAMES[f] for f in feats])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(axis="y", length=0)
        for _, fs in GROUPS:
            ax.axhspan(ypos[fs[0]] - 0.5, ypos[fs[-1]] + 0.5, color="#f4f3ee", zorder=0, lw=0)
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(axes[0].transAxes, axes[0].transData)
    for gname, fs in GROUPS:
        axes[0].text(-0.36, (ypos[fs[0]] + ypos[fs[-1]]) / 2, gname, rotation=90, ha="center",
                     va="center", fontsize=7, color=INK2, transform=tr, clip_on=False)
    # legends: dot size (share) and colour (tuning strength)
    lax = fig.add_axes([0.9, 0.3, 0.1, 0.4])
    lax.axis("off")
    for i, sh in enumerate((0.1, 0.3, 0.6)):
        lax.scatter([0.2], [1 - i * 0.22], s=260 * sh, color=SEQ(0.6), edgecolor="white", lw=0.4)
        lax.text(0.45, 1 - i * 0.22, f"{sh:.0%}", va="center", fontsize=6.5, color=INK2)
    lax.text(0.0, 1.17, "cells tuned", fontsize=6.8, color=INK)
    lax.set_xlim(0, 1)
    lax.set_ylim(0.2, 1.2)
    cax = fig.add_axes([0.91, 0.08, 0.012, 0.16])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    fig.text(0.9, 0.27, "tuning η²", fontsize=6.8, color=INK)
    steps, vis = [], set()
    for k in range(4):
        vis = vis | by_cond[k]
        steps.append(set(vis))
    save_steps(fig, steps, "tuning_matrix")
    plt.close(fig)

def tuning_curves(features=("speed", "turn")):
    fig, axes = plt.subplots(len(features), 4, figsize=(6.15, 2.85), sharey="row")
    fig.subplots_adjust(left=0.1, right=0.995, top=0.84, bottom=0.14, wspace=0.12, hspace=0.62)
    by_cond = [set() for _ in range(4)]
    for j, drug in enumerate(drugs):
        c = load(drug)
        bins = json.load(open(f"{res_dir}/bins_{drug.lower()}.json"))
        conds = conditions(drug)
        for i, feat in enumerate(features):
            e = np.asarray(bins[feat])
            centres = (e[:-1] + e[1:]) / 2
            for g, spn in enumerate(("D1", "D2")):
                ax = axes[i, 2 * j + g]
                ins = ax.inset_axes([0.22, 0.5, 0.28, 0.42] if feat == "speed" else
                                    [0.37, 0.5, 0.28, 0.42])      # share of tuned cells per condition
                ins.set_xlim(-0.6, 3.6)
                ins.set_ylim(0, 0.75)
                ins.set_xticks([])
                ins.set_yticks([0, 0.5], ["0", "50%"])
                ins.tick_params(axis="y", labelsize=5, length=1.5, pad=1)
                for sp in ("top", "right"):
                    ins.spines[sp].set_visible(False)
                ins.set_title("tuned", fontsize=5.5, pad=1, color=INK2, loc="left")
                ins.patch.set_facecolor("white")
                ins.set_zorder(4)
                for k, cond in enumerate(conds):
                    sh = c[(c.spn == spn) & (c.condition == cond) & (c.feature == feat)].tuned.mean()
                    by_cond[k] |= set(ins.bar([k], [sh], color=COND_COL[k], width=0.75).patches)
                    d = c[(c.spn == spn) & (c.condition == cond) & (c.feature == feat) & c.tuned]
                    C = d[[f"c{b}" for b in range(8)]].values
                    C = C / np.maximum(C.mean(1, keepdims=True), 1e-9)
                    d = d.assign(**{f"n{b}": C[:, b] for b in range(8)})
                    per = d.groupby("mouse").filter(lambda x: len(x) >= min_cells).groupby("mouse")[
                        [f"n{b}" for b in range(8)]].mean()
                    if len(per) < 2:
                        continue
                    m, se = per.mean().values, per.std().values / np.sqrt(len(per))
                    l, = ax.plot(np.arange(8), m, color=COND_COL[k], lw=1.4, marker="o", ms=2.5)
                    f_ = ax.fill_between(np.arange(8), m - se, m + se, color=COND_COL[k], alpha=0.18, lw=0)
                    by_cond[k] |= {l, f_}
                ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
                if feat == "speed":
                    lab = [f"{10 ** v - 0.5:.1f}" if 10 ** v - 0.5 < 10 else f"{10 ** v - 0.5:.0f}" for v in centres]
                else:
                    lab = [f"{5 * np.sinh(v):.0f}" for v in centres]
                ax.set_xticks([0, 3.5, 7], [lab[0], "", lab[7]])
                ax.set_xlabel({"speed": "speed bin (cm/s)", "turn": "turning bin (°/s)"}[feat], labelpad=1,
                              fontsize=7)
                if i == 0:
                    ax.set_title(f"{drug[:4]}. cohort, {spn}", loc="left", fontsize=7.5)
                if 2 * j + g == 0:
                    ax.set_ylabel("rate / mean rate")
                ax.yaxis.grid(True, color=GRID, lw=0.5)
                ax.set_axisbelow(True)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
    hs = [plt.Line2D([], [], color=COND_COL[k], lw=1.4, marker="o", ms=3) for k in range(4)]
    legs = [fig.legend(hs[:k + 1], COND_SHORT[:k + 1], loc="upper center", ncol=4, frameon=False,
                       bbox_to_anchor=(0.55, 1.0), columnspacing=1.4) for k in range(4)]
    steps, vis = [], set()
    for k in range(4):
        vis = (vis - set(legs)) | by_cond[k] | {legs[k]}
        steps.append(set(vis))
    save_steps(fig, steps, "tuning_curves")
    plt.close(fig)

bins_dir = os.environ.get("BINS_DIR", "/home/rgast/data/parker_data/spn_feature_bins")
BIN_ORDER = {"speed": ["rest", "slow", "fast"], "turning": ["left", "straight", "right"]}
BIN_LABEL = {"rest": "rest\n< 0.5 cm/s", "slow": "slow\n0.5-3 cm/s", "fast": "fast\n≥ 3 cm/s",
             "left": "left turn\n> 30°/s", "straight": "straight\n≤ 30°/s", "right": "right turn\n> 30°/s"}

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

def bins_figure(feature):
    """Mean rate and SD of log rate per bin of `feature`, per condition, D1 / D2, both cohorts."""
    rd = pd.concat([pd.read_csv(f"{bins_dir}/readouts_{d.lower()}.csv") for d in drugs], ignore_index=True)
    st = pd.read_csv(f"{bins_dir}/stats.csv")
    rd, st = rd[rd.feature == feature], st[st.feature == feature]
    bins = BIN_ORDER[feature]
    fig, axes = plt.subplots(2, 6, figsize=(6.6, 2.85), sharey="row")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.73, bottom=0.08, wspace=0.1, hspace=0.32)
    dx, gap = 0.75, 0.45
    xpos = lambda g, k: g * (4 * dx + gap) + k * dx
    by_cond = [set() for _ in range(4)]
    rows = [("mean_log_rate", "mean rate (Hz)"), ("sd_log_rate", "SD of log$_{10}$ rate")]
    for r_, (col, ylab) in enumerate(rows):
        lo, hi = rd[col].min(), rd[col].max()
        for j, drug in enumerate(drugs):
            conds = conditions(drug)
            for b, bn in enumerate(bins):
                ax = axes[r_, 3 * j + b]
                for g, spn in enumerate(("D1", "D2")):
                    p = rd[(rd.drug == drug) & (rd.bin == bn) & (rd.spn == spn)].pivot_table(
                        index="mouse", columns="condition", values=col).reindex(columns=conds)
                    xs = [xpos(g, k) for k in range(4)]
                    for k, cond in enumerate(conds):
                        v = p[cond].dropna().values
                        if not len(v):
                            continue
                        a1 = ax.scatter(np.full(len(v), xs[k]), v, s=4, color=COND_COL[k], alpha=0.45, lw=0)
                        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
                        a2 = ax.errorbar(xs[k], m, se, color=COND_COL[k], lw=1.1)
                        a3 = ax.scatter([xs[k]], [m], s=16, color=COND_COL[k], edgecolor="white", lw=0.6, zorder=3)
                        arts = {a1, a3, *a2.lines[0:1], *a2.lines[2]}
                        s_ = st[(st.drug == drug) & (st.bin == bn) & (st.spn == spn) & (st.readout == col)]
                        if k > 0:
                            q = s_[s_.contrast == ["", "Amph - Control", "drug - Control", "A+drug - Control"][k]]
                            if len(q) and stars(q.p_fdr.iloc[0]):
                                arts.add(ax.text(xs[k], hi + 0.04 * (hi - lo), stars(q.p_fdr.iloc[0]), ha="center",
                                                 fontsize=4.8, color=INK, va="bottom"))
                        if k == 3:
                            q = s_[s_.contrast == "A+drug - Amph"]
                            if len(q) and q.p_fdr.iloc[0] < 0.05:
                                n_ = len(stars(q.p_fdr.iloc[0]))
                                arts.add(ax.text(xs[k], hi + 0.2 * (hi - lo), "#" * n_, ha="center", fontsize=4.8,
                                                 color=COND_COL[3], va="bottom"))
                        by_cond[k] |= arts
                ax.set_xticks([xpos(0, 1.5), xpos(1, 1.5)], ["D1", "D2"] if r_ == 1 else ["", ""], fontsize=6.5)
                ax.tick_params(axis="x", length=0)
                ax.set_xlim(-0.4, xpos(1, 3) + 0.4)
                ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.36 * (hi - lo))
                ax.yaxis.grid(True, color=GRID, lw=0.5)
                ax.set_axisbelow(True)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if r_ == 0:
                    ax.set_title(BIN_LABEL[bn], fontsize=6.5, loc="center")
                if 3 * j + b == 0:
                    ax.set_ylabel(ylab, fontsize=7)
        if col == "mean_log_rate":
            yt = [t for t in (0.02, 0.05, 0.1, 0.2) if lo - 0.05 <= np.log10(t) <= hi + 0.3]
            axes[r_, 0].set_yticks(np.log10(yt), [f"{t:g}" for t in yt])
    for j, drug in enumerate(drugs):
        x0 = axes[0, 3 * j].get_position().x0
        x1 = axes[0, 3 * j + 2].get_position().x1
        fig.text((x0 + x1) / 2, 0.865, f"{drug} cohort", ha="center", fontsize=8, fontweight="bold", color=INK)
    hs = [plt.Line2D([], [], color=COND_COL[k], marker="o", lw=1.1, ms=3.5) for k in range(4)]
    legs = [fig.legend(hs[:k + 1], COND_SHORT[:k + 1], loc="upper center", ncol=4, frameon=False,
                       bbox_to_anchor=(0.54, 1.0), fontsize=6.8, columnspacing=1.4, handlelength=1.2)
            for k in range(4)]
    fig.text(0.995, 0.965, "* vs Control   # Amph. + drug vs Amph.", ha="right", fontsize=5.8, color=INK2)
    steps, vis = [], set()
    for k in range(4):
        vis = (vis - set(legs)) | by_cond[k] | {legs[k]}
        steps.append(set(vis))
    save_steps(fig, steps, f"bins_{feature}")
    plt.close(fig)


if __name__ == "__main__":
    tuning_matrix()
    tuning_curves()
    for f in ("speed", "turning"):
        bins_figure(f)
