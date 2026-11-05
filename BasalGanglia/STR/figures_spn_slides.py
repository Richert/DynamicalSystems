"""Slide figure of SPN firing-rate statistics per drug condition, behaviour-independent and in
example behaviours (from analysis_spn_behavior_readouts.py, readouts_<A_DRUG>.csv).

  spn_rate_stats   rows: mean and SD over cells of the log10 event rate (equal time and 50
                   cells per draw, 50 draws); columns: whole session (random 5 s windows) and
                   example behaviour labels; D1 and D2 mice; dots: recordings (one per mouse
                   and condition), lines: same mouse; mean +- SEM; brackets: mixed models
                   (analysis_spn_behavior_stats.py; mean: cell-level log rate, mouse and
                   recording random effects; SD: mouse random effect), BH-FDR, shown with >=
                   min_mice mice in both conditions

  conda run -n parker python figures_spn_slides.py
"""
import os
import itertools
import numpy as np
import pandas as pd
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
min_mice = 3
mixed_col = {"mean_log_rate": "log_rate", "sd_log_rate": "sd_log_rate"}
contrast_of = {("Control", "Amphetamine"): "Amph - Control",
               ("Control", f"A + Clozapine {a_drug}"): "A+Cloz - Control",
               ("Amphetamine", f"A + Clozapine {a_drug}"): "A+Cloz - Amph"}
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

def panel(ax, d, col, st):
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
            m = st[(st.readout == mixed_col[col]) & (st.spn == spn) &
                   (st.contrast == contrast_of[(conditions[i], conditions[j])])]
            if len(m) and m.n_mice_a.iloc[0] >= min_mice and m.n_mice_b.iloc[0] >= min_mice \
                    and stars(m.p_fdr.iloc[0]):
                tests.append((xs[i], xs[j], stars(m.p_fdr.iloc[0])))
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

def slide_rate_stats(rd, st):
    budget = rd.groupby("label")["T"].first()
    fig, axes = plt.subplots(2, len(columns), figsize=(6.15, 2.85), sharey="row")
    fig.subplots_adjust(left=0.1, right=0.995, top=0.8, bottom=0.08, wspace=0.12, hspace=0.32)
    for r, (col, ylab) in enumerate(rows):
        lo, hi = rd[rd.label.isin(columns)][col].quantile([0.0, 1.0])
        step = 0.09 * (hi - lo)
        for k, lab in enumerate(columns):
            ax = axes[r, k]
            brackets(ax, panel(ax, rd[rd.label == lab], col, st[st.label == lab]), step)
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
        axes[r, 0].set_ylim(lo - 0.05 * (hi - lo), hi + 0.62 * (hi - lo))
    x =axes[0, 0].get_position().x1 + 0.5 * (axes[0, 1].get_position().x0 - axes[0, 0].get_position().x1)
    fig.add_artist(plt.Line2D([x, x], [0.06, 0.88], color=MUTED, lw=0.6, ls=(0, (2, 2))))
    handles = [plt.Line2D([], [], color=cond_col[c], marker="o", ms=4.5, lw=1.3,
                          markeredgecolor="white", label=cond_short[c]) for c in conditions]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.55, 1.005), handlelength=1.6, columnspacing=1.6)
    save(fig, "spn_rate_stats")

def slide_tracked(tc, ts, labels=("whole session", "walk", "straight run")):
    """Within-cell Control vs Amphetamine log rates of tracked cells (Vehicle days): one square
    panel per behaviour, D1 (green) and D2 (purple) cells overlaid, mouse medians on top."""
    SPN_COL = {"D1": "#1f8a70", "D2": "#8a4fb5"}
    MED_COL = {"D1": "#0e4d3e", "D2": "#4c2470"}
    fig, axes = plt.subplots(1, len(labels), figsize=(6.15, 2.85), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.83, bottom=0.17, wspace=0.12)
    lim = (-2.55, 0.15)
    for k, lab in enumerate(labels):
        ax = axes[k]
        ax.plot(lim, lim, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_aspect("equal")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        spines(ax)
        ax.tick_params(axis="x", length=2.5)
        ax.set_title("all behaviour" if lab == "whole session" else lab, loc="left", fontsize=8.5,
                     fontweight="bold", color=INK)
        ax.set_xlabel("Control: log$_{10}$ rate (Hz)")
        if k == 0:
            ax.set_ylabel("Amph.: log$_{10}$ rate (Hz)")
        lines = []
        for spn in groups:
            d = tc[(tc.label == lab) & (tc.spn == spn)]
            s_ = ts[(ts.label == lab) & (ts.spn == spn) & (ts.readout == "d_rate")]
            if d.mouse.nunique() < 3 or not len(s_) or not np.isfinite(s_.p.iloc[0]):
                lines.append((spn, f"{spn}: too little Control time"))
                continue
            ax.scatter(d.ctrl_rate, d.amph_rate, s=2, color=SPN_COL[spn], alpha=0.16, lw=0,
                       rasterized=True, zorder=2)
            m = d.groupby("mouse")[["ctrl_rate", "amph_rate"]].median()
            ax.scatter(m.ctrl_rate, m.amph_rate, s=16, color=MED_COL[spn], edgecolor="white", lw=0.6,
                       zorder=4)
            s_ = s_.iloc[0]
            lines.append((spn, f"{spn}: Δ {s_.estimate:+.2f} {stars(s_.p_fdr) or 'n.s.'}, "
                               f"{s_.share_down:.0%} lower"))
        for i, (spn, t) in enumerate(lines):
            ax.text(0.03, 0.97 - 0.085 * i, t, transform=ax.transAxes, ha="left", va="top", fontsize=6,
                    color=MED_COL[spn])
    handles = [plt.Line2D([], [], color=SPN_COL[s_], marker="o", lw=0, ms=3.5, alpha=0.6, label=f"{s_} cell")
               for s_ in groups] + \
              [plt.Line2D([], [], color=INK2, marker="o", lw=0, ms=5, markeredgecolor="white",
                          label="mouse median"),
               plt.Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 2)), label="no change")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.54, 1.0),
               handlelength=1.4, columnspacing=1.4)
    save(fig, "spn_tracked_cells")

def slide_rate_pdfs(cv, st):
    """Whole-session per-cell event rates (all frames), D1 and D2 mice, per drug condition:
    distributions over cells and medians per mouse; stars: cell-level mixed models
    (stats_mixed_<A_DRUG>.csv, whole session)."""
    w = cv[cv.label == "whole session"]
    fig = plt.figure(figsize=(6.0, 2.6))
    gs = fig.add_gridspec(1, 7, width_ratios=[1.6, 0.3, 0.62, 0.3, 1.6, 0.3, 0.62], wspace=0.05,
                          left=0.085, right=0.99, top=0.8, bottom=0.18)
    med = w.groupby(["spn", "mouse", "condition"]).log_rate.median()
    mlo, mhi = med.min() - 0.08, med.max() + 0.32
    bins = np.linspace(-2.45, 0.35, 29)
    for g, spn in enumerate(groups):
        ax = fig.add_subplot(gs[4 * g])
        for c in conditions:
            v = w[(w.spn == spn) & (w.condition == c)].log_rate.clip(bins[0], bins[-1])
            h, _ = np.histogram(v, bins=bins)
            ax.stairs(h / h.sum(), bins, color=cond_col[c], lw=1.5)
            ax.stairs(h / h.sum(), bins, color=cond_col[c], fill=True, alpha=0.08)
        ax.set_xticks(np.log10([0.01, 0.03, 0.1, 0.3, 1]), ["0.01", "0.03", "0.1", "0.3", "1"])
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("event rate per cell (Hz)")
        ax.set_ylabel("share of cells" if g == 0 else None)
        ax.set_title(f"{spn} SPNs", loc="left", fontweight="bold", fontsize=9)
        n = w[w.spn == spn].groupby("condition").recording.nunique()
        ax.text(0.98, 0.97, f"{w[w.spn == spn].cell.size:,} cells\n{n.max()} mice".replace(",", " "),
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
        spines(ax)
        # medians per mouse
        ax = fig.add_subplot(gs[4 * g + 2])
        p = w[w.spn == spn].groupby(["mouse", "condition"]).log_rate.median().unstack().reindex(columns=conditions)
        for _, v in p.iterrows():
            ok = v.notna().values
            ax.plot(np.arange(3)[ok], v.values[ok], color=GRID, lw=0.6, zorder=1)
        for ci, c in enumerate(conditions):
            v = p[c].dropna().values
            ax.scatter(np.full(len(v), ci), v, s=8, color=cond_col[c], alpha=0.6, lw=0, zorder=2)
            ax.scatter([ci], [np.median(v)], s=28, color=cond_col[c], edgecolor="white", lw=0.8, zorder=3)
        top = p.max().max()
        for k, (cname, (i, j)) in enumerate((("Amph - Control", (0, 1)), ("A+Cloz - Amph", (1, 2)),
                                             ("A+Cloz - Control", (0, 2)))):
            s_ = st[(st.readout == "log_rate") & (st.label == "whole session") & (st.spn == spn) &
                    (st.contrast == cname)]
            if len(s_) and stars(s_.p_fdr.iloc[0]):
                y = top + 0.06 + (0.12 if k == 2 else 0)
                ax.plot([i + 0.08, i + 0.08, j - 0.08, j - 0.08], [y - 0.02, y, y, y - 0.02], color=INK2, lw=0.6)
                ax.text((i + j) / 2, y - 0.01, stars(s_.p_fdr.iloc[0]), ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(mlo, mhi)
        yt = [t for t in (0.03, 0.05, 0.1, 0.2) if mlo <= np.log10(t) <= mhi - 0.2]
        ax.set_yticks(np.log10(yt), [f"{t:g}" for t in yt])
        ax.tick_params(axis="y", labelsize=6)
        ax.set_ylabel("Hz", fontsize=6.5, labelpad=1)
        ax.set_title("mouse medians", loc="left", fontsize=6.5, color=INK2)
        spines(ax)
    handles = [plt.Line2D([], [], color=cond_col[c], lw=1.5, label=cond_short[c]) for c in conditions]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.6)
    save(fig, "spn_rate_pdfs")

exp_dir = os.environ.get("EXP_DIR", "/home/rgast/data/parker_data/spn_behavior_expected")

def slide_expected(res, st, readout="spikes", drugs=None, name=None):
    """Drug effect on the session activity split into the part expected from the behaviour
    change and the part beyond it (analysis_spn_behavior_expected.py); dots: mice.
    drugs: conditions shown (default: Amphetamine and A + drug high)."""
    comps = [("total", "total"), ("behaviour", "expected"), ("beyond", "beyond")]
    drugs = drugs or conditions[1:]
    dcol = {d: cond_col.get(d, "#8a4fb5" if not d.startswith(("A +", "Amph")) else "#e34948")
            for d in drugs}
    dshort = {d: cond_short.get(d, d.replace("Amphetamine", "Amph.").replace("Haloperidol", "Halo.")
                                 .replace("Clozapine", "Cloz.")) for d in drugs}
    span = 4.6
    fig, axes = plt.subplots(1, 2, figsize=(6.15 if len(drugs) < 3 else 7.4, 2.85), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.78, bottom=0.2, wspace=0.08)
    lo = min(-0.5, res[res.readout == readout][["total", "behaviour", "beyond"]].min().min() - 0.05)
    hi = max(0.62, res[res.readout == readout][["total", "behaviour", "beyond"]].max().max() + 0.12)
    for a_, spn in zip(axes, groups):
        for di, drug in enumerate(drugs):
            d = res[(res.readout == readout) & (res.spn == spn) & (res.drug == drug)]
            xs = [di * span + 1.35 * k for k in range(3)]
            for _, r_ in d.iterrows():
                a_.plot(xs, [r_.total, r_.behaviour, r_.beyond], color=GRID, lw=0.5, zorder=1)
            for k, (comp, _) in enumerate(comps):
                v = d[comp].values
                face = dcol[drug] if comp != "behaviour" else "white"
                edge = dcol[drug]
                a_.scatter(np.full(len(v), xs[k]), v, s=7, color=dcol[drug], alpha=0.4, lw=0, zorder=2)
                a_.errorbar(xs[k] + 0.22, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color=edge, lw=1.2, zorder=3)
                a_.scatter([xs[k] + 0.22], [v.mean()], s=34, facecolor=face, edgecolor=edge, lw=1.3,
                           zorder=4, marker="D" if comp == "beyond" else "o")
                s_ = st[(st.readout == readout) & (st.spn == spn) & (st.drug == drug) & (st.component == comp)]
                if len(s_) and stars(s_.p.iloc[0]):
                    a_.text(xs[k] + 0.22, hi - 0.07, stars(s_.p.iloc[0]), ha="center", fontsize=7, color=INK)
            a_.text(xs[1] + 0.1, -0.15, dshort[drug], ha="center", fontsize=7.2, color=dcol[drug],
                    fontweight="bold", transform=a_.get_xaxis_transform())
        a_.axhline(0, color=MUTED, lw=0.7)
        a_.set_xticks([di * span + 1.35 * k + 0.1 for di in range(len(drugs)) for k in range(3)],
                      [c[1] for _ in drugs for c in comps], fontsize=6.5)
        a_.tick_params(axis="x", length=0)
        a_.set_ylim(lo, hi)
        a_.set_xlim(-0.6, span * (len(drugs) - 1) + 3.4)
        a_.set_title(f"{spn} SPNs", loc="left", fontweight="bold", fontsize=9)
        a_.yaxis.grid(True, color=GRID, lw=0.5)
        a_.set_axisbelow(True)
        spines(a_)
    axes[0].set_ylabel("log$_{10}$ activity ratio\nto Control")
    ax2 = axes[1].twinx()                                  # % change scale
    ax2.set_ylim(axes[1].get_ylim())
    ticks = [t for t in (-80, -60, -40, -20, 0, 50, 100, 200) if lo <= np.log10(1 + t / 100) <= hi]
    ax2.set_yticks(np.log10(1 + np.array(ticks) / 100), [f"{t:+d} %" if t else "0 %" for t in ticks],
                   fontsize=6.5)
    ax2.spines["top"].set_visible(False)
    handles = [plt.Line2D([], [], color=INK2, marker="o", lw=0, ms=5, label="total: observed change"),
               plt.Line2D([], [], color=INK2, marker="o", lw=0, ms=5, mfc="white",
                          label="expected from the behaviour change"),
               plt.Line2D([], [], color=INK2, marker="D", lw=0, ms=4.5, label="beyond behaviour")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.4, handletextpad=0.3)
    save(fig, name or f"spn_expected_{readout}")

def save_steps(fig, steps, name):
    """Animation frames (see figures_raw_pipeline.save_steps): <name>_s<k>.pdf/.png."""
    managed = set().union(*steps)
    for k, vis in enumerate(steps, 1):
        for art in managed:
            art.set_visible(art in vis)
        fig.savefig(f"{out_dir}/{name}_s{k}.pdf")
        fig.savefig(f"{out_dir}/{name}_s{k}.png", dpi=150)
    for art in managed:
        art.set_visible(True)
    print(f"saved {out_dir}/{name}_s1..{len(steps)}")

def slide_rate_pdfs_steps(cv, rd, st):
    """Rate distributions (D1, D2) built up per condition, then the mean and the SD of the log
    rate per mouse (whole session, equal time / 50 cells per draw; mixed-model stars)."""
    w = cv[cv.label == "whole session"]
    fig = plt.figure(figsize=(6.6, 2.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.1, 1.48, 1.48], wspace=0.38, left=0.09,
                          right=0.99, top=0.8, bottom=0.2)
    bins = np.linspace(-2.45, 0.35, 29)
    hist = {c: set() for c in conditions}
    for g, spn in enumerate(groups):
        ax = fig.add_subplot(gs[g])
        for c in conditions:
            v = w[(w.spn == spn) & (w.condition == c)].log_rate.clip(bins[0], bins[-1])
            h, _ = np.histogram(v, bins=bins)
            hist[c] |= {ax.stairs(h / h.sum(), bins, color=cond_col[c], lw=1.5),
                        ax.stairs(h / h.sum(), bins, color=cond_col[c], fill=True, alpha=0.08)}
        ax.set_xticks(np.log10([0.01, 0.1, 1]), ["0.01", "0.1", "1"])
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("event rate per cell (Hz)")
        ax.set_ylabel("share of cells" if g == 0 else None)
        ax.set_title(f"{spn} SPNs", loc="left", fontweight="bold", fontsize=9)
        spines(ax)
    quant = []
    r0 = rd[rd.label == "whole session"]
    s0 = st[st.label == "whole session"]
    for j, (col, lab) in enumerate((("mean_log_rate", "mean rate (Hz)"), ("sd_log_rate", "SD of log$_{10}$ rate"))):
        ax = fig.add_subplot(gs[2 + j])
        lo, hi = r0[col].min(), r0[col].max()
        step = 0.075 * (hi - lo)
        brackets(ax, panel(ax, r0, col, s0), step)
        ax.set_xticks([xpos(g, 1) for g in range(len(groups))], groups)
        ax.set_xlim(xpos(0, 0) - 0.45, xpos(1, len(conditions) - 1) + 0.55)
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.45 * (hi - lo))
        if col == "mean_log_rate":
            yt = [t for t in (0.03, 0.05, 0.1, 0.2) if lo - 0.05 <= np.log10(t) <= hi + 0.3]
            ax.set_yticks(np.log10(yt), [f"{t:g}" for t in yt])
        ax.set_title(lab, loc="left", fontsize=8)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        spines(ax)
        quant.append({ax})
    handles = [plt.Line2D([], [], color=cond_col[c], lw=1.5, label=cond_short[c]) for c in conditions]
    legs = []
    for k in range(1, len(conditions) + 1):
        lg = fig.legend(handles=handles[:k], loc="upper left", ncol=3, frameon=False,
                        bbox_to_anchor=(0.07, 1.0), columnspacing=1.6)
        legs.append(lg)
    steps, vis = [], set()
    for k, c in enumerate(conditions):
        vis = (vis - set(legs)) | hist[c] | {legs[k]}
        steps.append(set(vis))
    for q in quant:
        vis = vis | q
        steps.append(set(vis))
    save_steps(fig, steps, "spn_rate_pdfs_anim")
    plt.close(fig)

# ---------------------------------------------------------------------------
# decoding (analysis_spn_decoding.py)
# ---------------------------------------------------------------------------

dec_dir = os.environ.get("DEC_DIR", "/home/rgast/data/parker_data/spn_decoding")
FEAT_GROUPS = [("movement", ["log_speed", "log_head_speed", "turn_asinh", "head_turn_asinh"]),
               ("posture", ["head_angle", "bend", "body_len", "ear_width", "hip_width"]),
               ("location", ["wall_dist", "wall_ahead"])]
FEAT_NAMES = {"log_speed": "speed", "log_head_speed": "head speed", "turn_asinh": "turning",
              "head_turn_asinh": "head turning", "head_angle": "head angle", "bend": "bend",
              "body_len": "body length", "ear_width": "ear width", "hip_width": "hip width",
              "wall_dist": "wall distance", "wall_ahead": "wall ahead"}
READ_COL = {"spikes": "#2a78d6", "trace": "#eb6834"}
READ_NAME = {"spikes": "spikes", "trace": "denoised traces"}

def decoding_table():
    d = pd.read_csv(f"{dec_dir}/decoding.csv")
    return d.groupby(["readout", "target", "day", "rec", "condition", "mouse", "spn"])[
        ["direct", "via", "ceiling", "null", "chance"]].mean().reset_index()

def spines(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

def slide_decoding_contrast(m, readouts=("spikes", "trace"), out_name="spn_decoding_contrast"):
    """Labels vs continuous features: (a) label decoding, direct vs via decoded features;
    (b) feature decoding, direct vs via decoded label probabilities; (c) share retained."""
    feats = [f for _, fs in FEAT_GROUPS for f in fs]
    fig = plt.figure(figsize=(6.15, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 2.15, 0.85], wspace=0.45, left=0.1,
                          right=0.99, top=0.8, bottom=0.27)
    # (a) labels
    ax = fig.add_subplot(gs[0])
    lab = m[m.target == "labels"]
    for i, ro in enumerate(readouts):
        d = lab[lab.readout == ro]
        x0 = i * 1.1
        for _, r_ in d.iterrows():
            ax.plot([x0, x0 + 0.45], [r_.via, r_.direct], color=GRID, lw=0.5, zorder=1)
        for dx_, col_, fill in ((0, "via", False), (0.45, "direct", True)):
            v = d[col_].values
            ax.scatter(np.full(len(v), x0 + dx_), v, s=6, color=READ_COL[ro], alpha=0.35, lw=0,
                       zorder=2)
            ax.scatter([x0 + dx_], [np.median(v)], s=30, zorder=3, lw=1.2,
                       facecolor=READ_COL[ro] if fill else "white", edgecolor=READ_COL[ro])
    ax.axhline(lab.chance.median(), color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(1.1 * (len(readouts) - 1) + 0.7, lab.chance.median(), "chance", fontsize=6, color=MUTED,
            ha="right", va="bottom")
    ax.set_xticks([x for i in range(len(readouts)) for x in (i * 1.1, i * 1.1 + 0.45)],
                  ["via feat.", "direct"] * len(readouts) if len(readouts) == 1 else ["via", "dir."] * len(readouts),
                  fontsize=6.5)
    ax.set_xlim(-0.3, 1.1 * (len(readouts) - 1) + 0.75)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("behaviour labels", loc="left")
    spines(ax)
    # (b) features
    ax = fig.add_subplot(gs[1])
    xs, ticks, x = {}, [], 0.0
    for gname, fs in FEAT_GROUPS:
        for f in fs:
            xs[f] = x
            x += 1
        ax.text(np.mean([xs[f] for f in fs]), 1.0, gname, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color=INK2)
        x += 0.6
    for ro, off in zip(readouts, (-0.17, 0.17) if len(readouts) == 2 else (0.0,)):
        for f in feats:
            d = m[(m.readout == ro) & (m.target == f)]
            for col_, fill in (("via", False), ("direct", True)):
                v = d[col_].values
                q = np.percentile(v, [25, 75])
                ax.plot([xs[f] + off] * 2, q, color=READ_COL[ro], lw=0.8, alpha=0.6 if fill else 0.35)
                ax.scatter([xs[f] + off], [np.median(v)], s=22 if fill else 18, lw=1.0, zorder=3,
                           facecolor=READ_COL[ro] if fill else "white", edgecolor=READ_COL[ro])
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks([xs[f] for f in feats], [FEAT_NAMES[f] for f in feats], rotation=45, ha="right",
                  fontsize=6.8)
    ax.set_xlim(-0.6, x - 0.4)
    ax.set_ylabel("correlation r (held out)")
    ax.set_title("pose features", loc="left", pad=12)
    spines(ax)
    # (c) share retained through the other representation
    ax = fig.add_subplot(gs[2])
    rows = []
    for (ro, day, rec), d in m.groupby(["readout", "day", "rec"]):
        l = d[d.target == "labels"].iloc[0]
        lab_ret = (l.via - l.chance) / (l.direct - l.chance) if l.direct > l.chance else np.nan
        f = d[d.target.isin(feats) & (d.direct > 0.1)]
        feat_ret = f.via.clip(lower=0).sum() / f.direct.sum() if len(f) else np.nan
        rows.append(dict(readout=ro, labels=lab_ret, features=feat_ret))
    rr = pd.DataFrame(rows)
    for i, (col_, name) in enumerate((("labels", "labels\nvia features"),
                                      ("features", "features\nvia labels"))):
        for j, ro in enumerate(readouts):
            v = rr[rr.readout == ro][col_].dropna().clip(-0.2, 1.5).values
            xx = i + ((j - 0.5) * 0.32 if len(readouts) == 2 else 0)
            ax.scatter(xx + np.random.default_rng(j).uniform(-0.06, 0.06, len(v)), v, s=6,
                       color=READ_COL[ro], alpha=0.35, lw=0)
            ax.scatter([xx], [np.median(v)], s=30, color=READ_COL[ro], edgecolor="white", lw=0.8,
                       zorder=3)
    ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks([0, 1], ["labels\nvia features", "features\nvia labels"], fontsize=6.8)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.05, 1.4)
    ax.set_ylabel("share of decoding retained")
    ax.set_title("detour", loc="left")
    spines(ax)
    handles = ([plt.Line2D([], [], color=READ_COL[ro], marker="o", lw=0, ms=4.5,
                           label=READ_NAME[ro]) for ro in readouts] if len(readouts) > 1 else []) + \
              [plt.Line2D([], [], color=INK2, marker="o", lw=0, ms=4.5, label="direct"),
               plt.Line2D([], [], color=INK2, marker="o", lw=0, ms=4.5, mfc="white",
                          label="via the other representation")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.53, 1.0),
               handletextpad=0.3, columnspacing=1.4)
    for x_, l_ in ((0.005, "a"), (0.27, "b"), (0.745, "c")):
        fig.text(x_, 0.85, l_, fontsize=10, fontweight="bold", color=INK)
    save(fig, out_name)
    print(rr.groupby("readout").median().round(3))

def decoding_stats(m, readout):
    """Mixed models of the direct decoding per feature (and labels): r ~ condition + (1 | mouse),
    D1 / D2 separately; BH-FDR over targets x SPN types x contrasts."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    import warnings
    out = []
    m = m.copy()
    lab = m.target == "labels"
    m.loc[lab, "direct"] = m.loc[lab, "direct"] - m.loc[lab, "chance"]   # labels: accuracy - chance
    for (t, spn), d in m[m.readout == readout].groupby(["target", "spn"]):
        d = d.copy()
        d["cond"] = pd.Categorical(d.condition, conditions)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(f"direct ~ C(cond, Treatment('Control'))", d, groups="mouse").fit(reml=True)
        names = list(fit.fe_params.index)
        v = lambda c: np.eye(len(names))[names.index(f"C(cond, Treatment('Control'))[T.{c}]")]
        for cname, vec in (("Amph - Control", v(conditions[1])),
                           ("A+Cloz - Amph", v(conditions[2]) - v(conditions[1])),
                           ("A+Cloz - Control", v(conditions[2]))):
            tt = fit.t_test(vec[None])
            out.append(dict(target=t, spn=spn, contrast=cname, estimate=float(tt.effect[0]),
                            p=float(tt.pvalue)))
    st = pd.DataFrame(out)
    st["p_fdr"] = multipletests(st.p, method="fdr_bh")[1]
    return st

def slide_decoding_conditions(m, readout="trace"):
    feats = [f for _, fs in FEAT_GROUPS for f in fs]
    st = decoding_stats(m, readout)
    st.to_csv(f"{dec_dir}/decoding_condition_stats_{readout}.csv", index=False)
    d0 = m[m.readout == readout]
    fig, axes = plt.subplots(2, 2, figsize=(6.15, 2.85), gridspec_kw=dict(width_ratios=[5.2, 0.75]),
                             sharey="col")
    fig.subplots_adjust(left=0.11, right=0.99, top=0.84, bottom=0.17, wspace=0.18, hspace=0.18)
    for r_, spn in enumerate(groups):
        ax = axes[r_, 0]
        xs, x = {}, 0.0
        for gname, fs in FEAT_GROUPS:
            for f in fs:
                xs[f] = x
                x += 1
            if r_ == 0:
                ax.text(np.mean([xs[f] for f in fs]), 1.02, gname, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7, color=INK2)
            x += 0.6
        for ci, c in enumerate(conditions):
            for f in feats:
                v = d0[(d0.target == f) & (d0.spn == spn) & (d0.condition == c)].direct.values
                xx = xs[f] + (ci - 1) * 0.24
                ax.scatter(np.full(len(v), xx), v, s=4, color=cond_col[c], alpha=0.35, lw=0)
                ax.errorbar(xx, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color=cond_col[c],
                            lw=1.1, capsize=0)
                ax.scatter([xx], [v.mean()], s=16, color=cond_col[c], edgecolor="white", lw=0.6,
                           zorder=3)
        top = d0[d0.target.isin(feats)].direct.max()
        for f in feats:                                   # Amph vs Control, A+Cloz vs Amph
            for k, (cname, x_off) in enumerate((("Amph - Control", -0.12), ("A+Cloz - Amph", 0.12))):
                s_ = st[(st.target == f) & (st.spn == spn) & (st.contrast == cname)]
                if len(s_) and stars(s_.p_fdr.iloc[0]):
                    ax.text(xs[f] + x_off, top + 0.04 + 0.07 * k, stars(s_.p_fdr.iloc[0]),
                            ha="center", fontsize=6,
                            color=cond_col[conditions[1]] if k == 0 else cond_col[conditions[2]])
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks([xs[f] for f in feats],
                      [FEAT_NAMES[f] for f in feats] if r_ == 1 else [], rotation=40, ha="right",
                      fontsize=6.8)
        ax.set_xlim(-0.6, x - 0.4)
        ax.set_ylim(min(-0.15, d0[d0.target.isin(feats)].direct.min() - 0.02), top + 0.2)
        ax.set_ylabel(f"{spn} mice\nfeature decoding r")
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        spines(ax)
        # labels (same readout)
        ax = axes[r_, 1]
        for ci, c in enumerate(conditions):
            dd = d0[(d0.target == "labels") & (d0.spn == spn) & (d0.condition == c)]
            v = (dd.direct - dd.chance).values
            ax.scatter(np.full(len(v), ci), v, s=4, color=cond_col[c], alpha=0.35, lw=0)
            ax.errorbar(ci, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color=cond_col[c], lw=1.1)
            ax.scatter([ci], [v.mean()], s=16, color=cond_col[c], edgecolor="white", lw=0.6, zorder=3)
        topl = (d0[(d0.target == "labels")].direct - d0[(d0.target == "labels")].chance).max()
        for k, (cname, xx) in enumerate((("Amph - Control", 0.5), ("A+Cloz - Amph", 1.5))):
            s_ = st[(st.target == "labels") & (st.spn == spn) & (st.contrast == cname)]
            if len(s_) and stars(s_.p_fdr.iloc[0]):
                ax.text(xx, 0.93, stars(s_.p_fdr.iloc[0]), ha="center", fontsize=6,
                        transform=ax.get_xaxis_transform(),
                        color=cond_col[conditions[1]] if k == 0 else cond_col[conditions[2]])
        ax.set_xticks([])
        ax.set_xlim(-0.6, 2.6)
        ax.axhline(0, color=MUTED, lw=0.6)
        if r_ == 0:
            ax.set_title("labels", loc="left", fontsize=7.5)
        if r_ == 1:
            ax.set_xlabel("acc. − chance", fontsize=6.8)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        spines(ax)
    handles = [plt.Line2D([], [], color=cond_col[c], marker="o", ms=4.5, lw=1.1,
                          markeredgecolor="white", label=cond_short[c]) for c in conditions]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.45, 1.0), columnspacing=1.6)
    fig.text(0.99, 0.965, "* Amph. vs Control", color=cond_col[conditions[1]], fontsize=6.5, ha="right")
    fig.text(0.99, 0.93, f"* {cond_short[conditions[2]]} vs Amph.", color=cond_col[conditions[2]],
             fontsize=6.5, ha="right")
    save(fig, f"spn_decoding_conditions_{readout}")
    pd.set_option("display.width", 200)
    print(st[st.p_fdr < 0.05].round(4).to_string(index=False))


if __name__ == "__main__":
    rd = pd.read_csv(f"{res_dir}/readouts_{a_drug}.csv")
    st_mixed = pd.read_csv(f"{res_dir}/stats_mixed_{a_drug}.csv")
    slide_rate_pdfs(pd.read_csv(f"{res_dir}/cell_values_{a_drug}.csv.gz"), st_mixed)
    slide_rate_stats(rd, st_mixed)
    slide_tracked(pd.read_csv(f"{res_dir}/tracked_cells.csv.gz"),
                  pd.read_csv(f"{res_dir}/stats_tracked.csv"))
    if os.path.isfile(f"{exp_dir}/expected_by_behaviour.csv"):
        for ro in ("spikes", "trace"):
            slide_expected(pd.read_csv(f"{exp_dir}/expected_by_behaviour.csv"),
                           pd.read_csv(f"{exp_dir}/expected_stats.csv"), ro)
    if os.path.isfile(f"{dec_dir}/decoding.csv"):
        m = decoding_table()
        slide_decoding_contrast(m)
        for ro in ("trace", "spikes"):
            slide_decoding_conditions(m, ro)
