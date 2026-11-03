"""Slide figure of SPN firing-rate statistics per drug condition, behaviour-independent and in
example behaviours (from analysis_spn_behavior_readouts.py, readouts_<A_DRUG>.csv).

  spn_rate_stats   rows: mean and SD over cells of the log10 event rate (equal time and 50
                   cells per draw, 50 draws); columns: whole session (random 5 s windows) and
                   example behaviour labels; D1 and D2 mice; dots: recordings (one per mouse
                   and condition), lines: same mouse; mean +- SEM; brackets: Wilcoxon
                   signed-rank test over mice (paired), uncorrected

  conda run -n parker python figures_spn_slides.py
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = os.environ.get("SLIDE_DIR", "/home/rgast/OneDrive/presentations/parker/images")
res_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/spn_behavior_states")
a_drug = os.environ.get("A_DRUG", "high")
columns = ["whole session", "rest", "walk", "scanning walk"]
conditions = ["Control", "Amphetamine", f"A + Clozapine {a_drug}"]
cond_short = {"Control": "Control", "Amphetamine": "Amph.",
              f"A + Clozapine {a_drug}": f"Amph. + Cloz. {a_drug}"}
cond_col = dict(zip(conditions, ["#2a78d6", "#eda100", "#e34948"]))
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7,
                     "legend.fontsize": 7.5, "pdf.fonttype": 42, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
                     "xtick.major.size": 0, "ytick.major.size": 2.5, "axes.linewidth": 0.7,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
rows = [("mean_log_rate", "mean of log$_{10}$ rate"), ("sd_log_rate", "SD of log$_{10}$ rate")]
groups = ["D1", "D2"]
dx, gap = 0.62, 0.75                       # spacing of conditions within / between D1 and D2

def save(fig, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.pdf")
    fig.savefig(f"{out_dir}/{name}.png", dpi=200)
    plt.close(fig)
    print(f"saved {out_dir}/{name}.pdf/.png")

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else None

def xpos(g, c):
    return g * (len(conditions) * dx + gap) + c * dx

def panel(ax, d, col):
    tops = []
    for g, spn in enumerate(groups):
        p = d[d.spn == spn].pivot_table(index="mouse", columns="condition", values=col)
        p = p.reindex(columns=conditions)
        xs = [xpos(g, c) for c in range(len(conditions))]
        for _, v in p.iterrows():                       # same mouse
            ok = v.notna().values
            ax.plot(np.array(xs)[ok], v.values[ok], color=GRID, lw=0.6, zorder=1)
        for c, cond in enumerate(conditions):
            v = p[cond].dropna().values
            if not len(v):
                continue
            ax.scatter(np.full(len(v), xs[c]), v, s=9, color=cond_col[cond], alpha=0.55,
                       lw=0, zorder=2)
            m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            ax.errorbar(xs[c] + 0.2, m, se, color=cond_col[cond], lw=1.3, capsize=0, zorder=3)
            ax.scatter([xs[c] + 0.2], [m], s=26, color=cond_col[cond], edgecolor="white",
                       lw=0.9, zorder=4)
            if len(v) < 4:
                ax.text(xs[c], np.nanmax(p.values) , f"n={len(v)}", ha="center", va="bottom",
                        fontsize=5.5, color=MUTED)
        tests = []
        for i, j in itertools.combinations(range(len(conditions)), 2):
            q = p[[conditions[i], conditions[j]]].dropna()
            if len(q) >= 5:
                pv = wilcoxon(q.iloc[:, 0], q.iloc[:, 1]).pvalue
                if stars(pv):
                    tests.append((xs[i], xs[j], stars(pv)))
        tops.append((np.nanmax(p.values), tests))
    return tops

def brackets(ax, tops, step):
    for top, tests in tops:
        levels = []                                        # greedy: non-overlapping share a level
        for x0, x1, s in sorted(tests, key=lambda t: t[1] - t[0]):
            lv = next((i for i, used in enumerate(levels)
                       if all(x0 >= b - 0.05 or x1 <= a + 0.05 for a, b in used)), len(levels))
            if lv == len(levels):
                levels.append([])
            levels[lv].append((x0 + 0.04, x1 - 0.04))
            y = top + (0.7 + 1.9 * lv) * step
            ax.plot([x0 + 0.04, x0 + 0.04, x1 - 0.04, x1 - 0.04],
                    [y - 0.25 * step, y, y, y - 0.25 * step], color=INK2, lw=0.6)
            ax.text((x0 + x1) / 2, y - 0.3 * step, s, ha="center", va="bottom", fontsize=7,
                    color=INK)

def slide_rate_stats(rd):
    budget = rd.groupby("label")["T"].first()
    fig, axes = plt.subplots(2, len(columns), figsize=(6.0, 3.35), sharey="row")
    fig.subplots_adjust(left=0.1, right=0.995, top=0.8, bottom=0.08, wspace=0.12, hspace=0.32)
    for r, (col, ylab) in enumerate(rows):
        lo, hi = rd[rd.label.isin(columns)][col].quantile([0.0, 1.0])
        step = 0.075 * (hi - lo)
        for k, lab in enumerate(columns):
            ax = axes[r, k]
            brackets(ax, panel(ax, rd[rd.label == lab], col), step)
            ax.set_xticks([xpos(g, 1) for g in range(len(groups))], groups if r == 1 else [])
            ax.set_xlim(xpos(0, 0) - 0.45, xpos(1, len(conditions) - 1) + 0.55)
            ax.yaxis.grid(True, color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ttl = "all behaviour" if lab == "whole session" else lab
                sub = (f"random {budget[lab]:.0f} s of the session" if lab == "whole session"
                       else f"{budget[lab]:.0f} s in the label")
                ax.set_title(ttl, loc="left", fontsize=8.5, fontweight="bold", color=INK, pad=12)
                ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=6.5, color=INK2)
            if k == 0:
                ax.set_ylabel(ylab)
        axes[r, 0].set_ylim(lo - 0.05 * (hi - lo), hi + 0.5 * (hi - lo))
    x =axes[0, 0].get_position().x1 + 0.5 * (axes[0, 1].get_position().x0 - axes[0, 0].get_position().x1)
    fig.add_artist(plt.Line2D([x, x], [0.06, 0.88], color=MUTED, lw=0.6, ls=(0, (2, 2))))
    handles = [plt.Line2D([], [], color=cond_col[c], marker="o", ms=4.5, lw=1.3,
                          markeredgecolor="white", label=cond_short[c]) for c in conditions]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.55, 1.005), handlelength=1.6, columnspacing=1.6)
    save(fig, "spn_rate_stats")


if __name__ == "__main__":
    rd = pd.read_csv(f"{res_dir}/readouts_{a_drug}.csv")
    slide_rate_stats(rd)
