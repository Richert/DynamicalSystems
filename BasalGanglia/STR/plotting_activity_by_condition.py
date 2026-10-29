"""Summary figures of the behaviour-independent SPN activity analysis, one per window
length T (outputs of analysis_activity_by_condition.py).

Per figure: rows D1 / D2; first column = per-cell log-rate distribution per condition
(Gaussian mixture on the active cells, scaled by the active share; silent share in the
label); further columns = recording readouts per condition (mice as connected grey
points, mean +/- SEM over mice, brackets for contrasts with q < 0.05 from the mixed
models).

Colours: validated 4-slot categorical set (blue, aqua, yellow, red; all-pairs CVD
dE >= 6.9 -> secondary encoding required: conditions also differ by x position / line
style and curves are direct-labelled).

  /home/rgast/conda/envs/arhmm/bin/python plotting_activity_by_condition.py [--permouse]
"""

import sys
import json
import numpy as np
import pandas as pd

in_dir = "/home/rgast/data/parker_data/behavior_independent"
drug = "Clozapine"
conditions = ["Control", drug, "Amphetamine", f"A + {drug}"]
colors = dict(zip(conditions, ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]))
styles = dict(zip(conditions, ["-", "--", ":", "-."]))
ink, ink2, grid_c = "#1f1f1e", "#5f5e5a", "#e6e5e0"
# per SPN type two rows: spike-based measures, then ISI summaries + trace-based measures
row_a = ["rate_dist", ("mean_log", "mean log$_{10}$ rate (Hz)"),
         ("sd_log", "SD of log$_{10}$ rate"), "isi_dist"]
row_b = [("isi_med_log", "median ISI (log$_{10}$ s)"), ("isi_burst", "share of ISIs 0.2-0.5 s"),
         ("corr_excess", "pairwise corr. of traces\n(minus shuffled)"),
         ("pr_ratio", "participation ratio of traces\n(real / shuffled)")]
suffix = "_permouse" if "--permouse" in sys.argv else ""

def stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*"

def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink2)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.yaxis.grid(True, color=grid_c, lw=0.8)
    ax.set_axisbelow(True)

tick_labels = {"Control": "Control", drug: drug, "Amphetamine": "Amphet-\namine",
               f"A + {drug}": f"A +\n{drug}"}

def distribution_panel(ax, dist, T_s, spn):
    from scipy.stats import norm
    floor = np.log10(1 / T_s)                     # one event per window (dequantised fit)
    xs = np.linspace(floor, 0.6, 400)
    d = dist[(dist.T_s == T_s) & (dist.spn == spn)]
    peaks = []
    for cond in conditions:
        row = d[d.condition == cond]
        if row.empty:
            continue
        r = row.iloc[0]
        m, s, w = (np.array(json.loads(r[c])) for c in ("means", "sds", "weights"))
        dens = (1 - r.silent) * sum(wi * norm.pdf(xs, mi, si) for mi, si, wi in zip(m, s, w))
        ax.plot(xs, dens, color=colors[cond], ls=styles[cond], lw=2,
                label=f"{cond}  (silent {r.silent:.0%})")
        peaks.append(dens.max())
    ax.set_xlim(floor, 0.6)
    ax.set_ylim(0, max(peaks) * 1.15 if peaks else 1)
    ax.set_xlabel("log$_{10}$ event rate (Hz), active cells (counts dequantised)",
                  fontsize=8, color=ink)
    ax.set_ylabel("density x active share", fontsize=8, color=ink)
    ax.legend(fontsize=7, frameon=False, loc="upper left", labelcolor=ink)
    style_axes(ax)

def isi_panel(ax, isi, T_s, spn):
    d = isi[(isi.T_s == T_s) & (isi.spn == spn)]
    for cond in conditions:
        g = d[d.condition == cond]
        if g.empty:
            continue
        ax.plot(g.isi_s, g.density, color=colors[cond], ls=styles[cond], lw=2,
                label=f"{cond}  (n = {g.n_isi.iloc[0]:,} ISIs)")
    for b in (0.2, 0.5):                          # burst ISI range
        ax.axvline(b, color=ink2, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.05, T_s)
    ax.set_xlabel("inter-spike interval (s, dequantised); within windows, so < T", fontsize=8, color=ink)
    ax.set_ylabel("density per log$_{10}$ unit", fontsize=8, color=ink)
    ax.legend(fontsize=7, frameon=False, loc="upper right", labelcolor=ink)
    style_axes(ax)

def readout_panel(ax, rec, stats, q, T_s, spn):
    d = rec[(rec.T_s == T_s) & (rec.spn == spn)]
    pm = d.groupby(["mouse", "condition"])[q].mean().unstack().reindex(columns=conditions)
    x = np.arange(len(conditions))
    for _, row in pm.iterrows():                               # mice
        ok = row.notna().values
        ax.plot(x[ok], row.values[ok], color="#c9c8c2", lw=0.8, zorder=1)
        ax.scatter(x[ok], row.values[ok], color="#c9c8c2", s=10, zorder=1)
    mean, sem = pm.mean(), pm.sem()
    for i, c in enumerate(conditions):                         # condition mean +/- SEM
        ax.errorbar(i, mean[c], yerr=sem[c], fmt="o", ms=7, color=colors[c], mec=ink,
                    mew=0.6, ecolor=ink, elinewidth=1.2, capsize=3, zorder=3)
    # significance brackets
    s = stats[(stats.T_s == T_s) & (stats.spn == spn) & (stats.quantity == q) & (stats.q_fdr < 0.05)]
    lo, hi = np.nanmin(pm.values), np.nanmax(pm.values)
    step, level = (hi - lo) * 0.09, hi + (hi - lo) * 0.06
    for t in s.sort_values("contrast").itertuples():
        a, b = t.contrast.split(" - ")
        i, j = conditions.index(a), conditions.index(b)
        ax.plot([i, i, j, j], [level - step * 0.3, level, level, level - step * 0.3],
                color=ink2, lw=0.9)
        ax.text((i + j) / 2, level, stars(t.q_fdr), ha="center", va="bottom", fontsize=8,
                color=ink)
        level += step
    ax.set_ylim(lo - (hi - lo) * 0.08, level + step * 0.3)
    ax.set_xticks(x, [tick_labels[c] for c in conditions], fontsize=7.5, color=ink)
    ax.set_xlim(-0.5, len(conditions) - 0.5)
    style_axes(ax)

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec = pd.read_csv(f"{in_dir}/recording_readouts{suffix}.csv")
    stats = pd.read_csv(f"{in_dir}/condition_contrasts{suffix}.csv")
    dist = pd.read_csv(f"{in_dir}/rate_distributions{suffix}.csv")
    isi = pd.read_csv(f"{in_dir}/isi_distributions{suffix}.csv")
    for T_s in sorted(rec.T_s.unique()):
        fig, axes = plt.subplots(4, 4, figsize=(18, 16), layout="constrained",
                                 gridspec_kw=dict(width_ratios=[1.5, 1, 1, 1.5]))
        for s, spn in enumerate(["D1", "D2"]):
            n_mice = rec[(rec.T_s == T_s) & (rec.spn == spn)].mouse.nunique()
            for r, row in enumerate([row_a, row_b]):
                for c, p in enumerate(row):
                    ax = axes[2 * s + r, c]
                    if p == "rate_dist":
                        distribution_panel(ax, dist, T_s, spn)
                        title = f"{spn} SPNs ({n_mice} mice) - rate distribution"
                    elif p == "isi_dist":
                        isi_panel(ax, isi, T_s, spn)
                        title = f"{spn}: ISI distribution"
                    else:
                        q, label = p
                        readout_panel(ax, rec, stats, q, T_s, spn)
                        ax.set_ylabel(label, fontsize=8.5, color=ink)
                        title = f"{spn}: {label.replace(chr(10), ' ')}"
                    ax.set_title(title, fontsize=10, color=ink, loc="left")
        fig.suptitle(f"SPN activity by drug condition, independent of behaviour  |  windows of "
                     f"T = {T_s} s, 50 per recording, 50 cells each  |  rates and ISIs from "
                     f"inferred spikes, correlation / PR from CNMF-E denoised traces\n"
                     f"grey: mice, coloured: mean ± SEM  |  brackets: mixed model value ~ "
                     f"condition + (1|mouse), BH-FDR q < 0.05 (* <0.05, ** <0.01, *** <0.001)",
                     fontsize=10.5, color=ink)
        out = f"{in_dir}/summary_T{T_s}{suffix}.png"
        fig.savefig(out, dpi=140, facecolor="white")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
