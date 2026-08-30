"""Composite figure: embedding-based behavioural states x SPN firing-rate dynamics.

Sibling of plotting_arhmm_figure.py, but for the delay-embedding pipeline
(analysis_embed_spn_states.py). Differences vs the HMM figure:

  Panel A : example velocity trace, time windows shaded by the
            *cluster* state (same colour code as panel B).
  Panel B : the 2-D t-SNE/Isomap embedding of the delay-embedded behaviour,
            colour-coded by the clustering outcome (speed-ordered states).
  Panels C/D (+ further drug rows) : selected dependent variables (center,
            slope, ...) for a selected number of cluster states, by condition.

Reuses analysis_embed_spn_states.py (label propagation, config) and
analysis_arhmm_spn_states.py (loading, readouts). Never re-fits: it loads the
exact model saved by analysis_embed_spn_states.py (embed_model.pkl).

  /home/rgast/conda/envs/arhmm/bin/python plotting_embed_figure.py
"""

import os
import pickle
import numpy as np
import analysis_arhmm_spn_states as A
import analysis_embed_spn_states as E

save_dir = A.save_dir
# EXACT model produced by analysis_embed_spn_states.py; this script never refits.
model_path = E.model_path
fs = A.fs
drugs = ["haloperidol"]
conditions = ["Control", "Amphetamine", "A + Drug"]
example_condition = "Amphetamine"
example_win_s = 100.0         # length of the example window (row 1)
# cluster states used for the bar+line panels (rows 1+), and the 2 dependent
# variables to plot (one panel each). vars: occupancy | slope | center | dim
panel_states = [0, 1, 2, 3]
panel_labels = ["s0", "s1", "s2", "s3"]
panel_vars = ["center", "inv_slope"]   # panel D = 1/slope (SPN rate heterogeneity)

# ---------------------------------------------------------------------------
# labelled sessions (load the saved embedding model, never re-fit)
# ---------------------------------------------------------------------------

def get_labelled():
    """Load the EXACT embedding model fitted by analysis_embed_spn_states.py and
    label every frame of the same sessions with it (kNN propagation + the saved
    speed-ordered remap)."""
    sessions = A.load_sessions()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run analysis_embed_spn_states.py first -- it "
            f"fits the embedding/clustering model and saves it here, and this figure "
            f"reuses that exact model.")
    m = pickle.load(open(model_path, "rb"))
    mu, sd, remap = m["mu"], m["sd"], m["remap"]
    E.embed_lags, E.embed_step = m["embed_lags"], m["embed_step"]
    for s in sessions:
        s["obs_z"] = (s["obs"] - mu) / sd
        s["feat"] = E.build_features(s)
    E.predict_raw(sessions, m["knn"])                 # raw cluster ids (-1 = low conf)
    lut = np.full(m["n_clusters"], -1, int)
    for old, new in remap.items():
        lut[old] = new
    for s in sessions:
        v = s["z"] >= 0
        z = np.full_like(s["z"], -1); z[v] = lut[s["z"][v]]; s["z"] = z
    state_speed = m["state_speed"]
    print(f"Loaded embedding model (K={len(state_speed)}, method={m['embed_method']}, "
          f"lags={m['embed_lags']}).")
    lag_of = A.animal_lags(sessions)
    return sessions, state_speed, lag_of, m

# ---------------------------------------------------------------------------
# row-2 summary metrics (shared with the HMM figure)
# ---------------------------------------------------------------------------

_ADRUG = {"haloperidol": "A+halo", "xanomeline": "A+xano", "MP10": "A+MP10"}

def cond5_of(condition, drug):
    if condition == "Control":
        return "Control"
    if condition == "Amphetamine":
        return "Amphetamine"
    return _ADRUG.get(drug)

def readouts_df(sessions, lag_of):
    """Per (session, state) occupancy + CDF slope/center/dim, with a 'cond5' label."""
    from pandas import DataFrame
    rng = np.random.default_rng(A.seed + 1)
    df = DataFrame(A.state_readouts(sessions, lag_of, rng))
    df["cond5"] = [cond5_of(c, d) for c, d in zip(df["condition"], df["drug"])]
    # 1/slope of the firing-rate CDF = spread of the log-rate distribution, i.e. a
    # measure of across-SPN firing-rate heterogeneity (higher = more heterogeneous).
    df["inv_slope"] = np.where(df["slope"] > 0, 1.0 / df["slope"], np.nan)
    return df

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def make_figure(sessions, state_speed, lag_of, model, rows_drugs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from scipy.stats import ttest_rel
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    import seaborn as sb

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    legend_fs = plt.rcParams["xtick.labelsize"]   # all legends match the tick labels
    cb = sb.color_palette("colorblind")
    cond_colors = {"Control": cb[0], "Amphetamine": cb[1], "A + Drug": cb[2]}
    K = len(state_speed)
    state_colors = sb.color_palette("Set2", K)
    df = readouts_df(sessions, lag_of)

    rows_drug = [d for d in rows_drugs if d in set(df["drug"])] or sorted(set(df["drug"]))[:1]
    n_rows = len(rows_drug)

    fig = plt.figure(figsize=(18, 4.6 + 3.8 * n_rows))
    gs = fig.add_gridspec(1 + n_rows, 4, height_ratios=[1.15] + [1.0] * n_rows,
                          hspace=0.5, wspace=0.55)

    def panel_letter(ax, letter, dx=-0.14):
        ax.text(dx, 1.05, letter, transform=ax.transAxes, fontsize=20,
                fontweight="bold", va="bottom", ha="right")

    # ---- ROW 0, A (cols 0-2): example trace shaded by cluster state ----
    ax = fig.add_subplot(gs[0, 0:3])
    Lwin = int(example_win_s * fs)
    target = list(panel_states)
    nsel = len(target)
    # search Control sessions for the window (of the requested duration) that
    # maximises the number of frames in the selected states while keeping their
    # distribution as uniform as possible: score = total_frames * (perplexity/nsel)
    best = None
    for s in (x for x in sessions if x["condition"] == "Control"):
        z = s["z"]; n = len(z)
        L = min(Lwin, n)
        starts = np.arange(n - L + 1)
        counts = np.zeros((nsel, len(starts)))
        for ki, k in enumerate(target):
            c = np.concatenate([[0], np.cumsum(z == k)])
            counts[ki] = c[starts + L] - c[starts]
        total = counts.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(total > 0, counts / total, 0.0)
            H = -np.where(p > 0, p * np.log(p), 0.0).sum(0)
        perplexity = np.exp(H)
        score = total * (perplexity / nsel)
        j = int(np.argmax(score))
        if best is None or score[j] > best[0]:
            best = (float(score[j]), s, int(starts[j]), float(total[j]), float(perplexity[j]))
    cand, i0 = best[1], best[2]
    z = cand["z"]; i1 = min(len(z), i0 + Lwin)
    print(f"Panel A: {cand['mouse']}/{cand['drug']}/Control — {best[3]:.0f} selected-state "
          f"frames, uniformity {best[4] / nsel:.2f} (perplexity {best[4]:.2f}/{nsel})")
    t = np.arange(i0, i1) / fs
    ax.plot(t, cand["speed"][i0:i1], color="black", lw=1.6)
    ymax = cand["speed"][i0:i1].max() * 1.05
    for k in panel_states:   # only shade the states selected for panels C+
        ax.fill_between(t, 0, ymax, where=(z[i0:i1] == k), color=state_colors[k],
                        alpha=0.35, step="mid", linewidth=0)
    ax.set(xlim=(t[0], t[-1]), ylim=(0, ymax), xlabel="time (s)", ylabel="velocity (cm/s)")
    state_handles = [Patch(facecolor=state_colors[k], alpha=0.5,
                           label=panel_labels[i]) for i, k in enumerate(panel_states)]
    ax.legend(handles=state_handles, ncol=len(panel_labels), fontsize=legend_fs,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    panel_letter(ax, "A", dx=-0.05)

    # ---- ROW 0, B (col 3): 2-D behaviour embedding coloured by cluster state ----
    ax = fig.add_subplot(gs[0, 3])
    emb2d, raw_labels, remap = model["emb2d"], model["emb_labels"], model["remap"]
    new = np.array([remap.get(int(l), -1) for l in raw_labels])
    order = np.argsort([np.sum(new == k) for k in range(K)])   # plot big clusters first
    for k in order:
        sel = new == k
        ax.scatter(emb2d[sel, 0], emb2d[sel, 1], s=3, color=state_colors[k],
                   alpha=0.5, linewidths=0, rasterized=True)
    ax.set(xlabel=f"{model['embed_method']}-1", ylabel=f"{model['embed_method']}-2",
           xticks=[], yticks=[])
    ax.set_title("behaviour embedding")
    emb_handles = [Line2D([0], [0], marker="o", linestyle="", color=state_colors[k],
                          label=panel_labels[i], markersize=7)
                   for i, k in enumerate(panel_states)]
    ax.legend(handles=emb_handles, fontsize=legend_fs, loc="best", framealpha=0.6,
              handletextpad=0.2, borderpad=0.3)
    panel_letter(ax, "B", dx=-0.3)

    # ---- ROWS 1+: boxen panels (rows = drugs) with state x condition ANOVA ----
    def per_animal(drug, var):
        d = df[(df["drug"] == drug) & (df["state"].isin(panel_states))
               & (df["condition"].isin(conditions))].dropna(subset=[var])
        return d.groupby(["mouse", "state", "condition"], as_index=False)[var].mean()

    def paired_p(pa, state, var, cA, cB):
        w = pa[pa["state"] == state].pivot_table(index="mouse", columns="condition", values=var)
        if cA in w.columns and cB in w.columns:
            sub = w[[cA, cB]].dropna()
            if len(sub) >= 3 and np.ptp(sub[cA] - sub[cB]) > 0:
                return float(ttest_rel(sub[cA], sub[cB]).pvalue)
        return np.nan

    def two_way_anova(pa, var):
        if pa["state"].nunique() < 2 or pa["condition"].nunique() < 2 or len(pa) < 8:
            return None
        try:
            aov = anova_lm(smf.ols(f"{var} ~ C(state)*C(condition)", data=pa).fit(), typ=2)
            return {"state": aov.loc["C(state)", "PR(>F)"],
                    "cond": aov.loc["C(condition)", "PR(>F)"],
                    "inter": aov.loc["C(state):C(condition)", "PR(>F)"]}
        except Exception:
            return None

    def stars(p):
        return "" if not np.isfinite(p) else \
            "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."

    def pstr(p):
        return "n/a" if not np.isfinite(p) else "<0.001" if p < 1e-3 else f"{p:.3g}"

    VAR_LABEL = {"occupancy": "occupancy (fraction)", "slope": "SPN CDF slope (1/$\\sigma$)",
                 "inv_slope": "SPN rate heterogeneity",
                 "center": "mean SPN rate (log$_{10}$)", "dim": "SPN dimensionality"}
    VAR_TITLE = {"occupancy": "state occupancy", "slope": "SPN CDF slope",
                 "inv_slope": "SPN CDF 1/slope",
                 "center": "SPN CDF center", "dim": "SPN dimensionality"}
    VAR_ALIAS = {"dimensionality": "dim", "firing rate slope": "slope",
                 "firing rate center": "center", "firing_rate_slope": "slope"}
    pvars = [VAR_ALIAS.get(v, v) for v in panel_vars]
    ncond = len(conditions)
    box_w = 0.7
    off = box_w / ncond

    for di, drug in enumerate(rows_drug):
        for vi, var in enumerate(pvars):
            ax = fig.add_subplot(gs[di + 1, 2 * vi:2 * vi + 2])
            pa = per_animal(drug, var)
            sb.boxenplot(data=pa, x="state", y=var, hue="condition", order=panel_states,
                         hue_order=conditions, palette=cond_colors, width=box_w,
                         linewidth=0.8, legend=False, ax=ax)
            vals = pa[var].to_numpy()
            lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
            rng0 = (hi - lo) or (abs(hi) or 1.0)
            ax.set_ylim(lo - 0.08 * rng0, hi + 0.35 * rng0)
            yr = ax.get_ylim()[1] - ax.get_ylim()[0]

            contrasts = []
            for si, st in enumerate(panel_states):
                for cA, cB in [("Control", "Amphetamine"), ("Amphetamine", "A + Drug")]:
                    p = paired_p(pa, st, var, cA, cB)
                    xc = si + ((conditions.index(cA) + conditions.index(cB)) / 2 - (ncond - 1) / 2) * off
                    contrasts.append(dict(p=p, xc=xc))
            minp = min((c["p"] for c in contrasts if np.isfinite(c["p"])), default=np.inf)
            star_y = ax.get_ylim()[1] - 0.16 * yr
            for c in contrasts:
                s = stars(c["p"])
                if not s:
                    continue
                hot = np.isfinite(c["p"]) and c["p"] == minp and c["p"] < 0.05
                ax.text(c["xc"], star_y, s, ha="center", va="center", zorder=6,
                        fontsize=13 if s != "n.s." else 8,
                        fontweight="bold" if s != "n.s." else "normal",
                        color=("crimson" if hot else ("0.1" if s != "n.s." else "0.55")))

            aov = two_way_anova(pa, var)
            if aov:
                print(f"[ANOVA] {drug} / {var}:  state p={pstr(aov['state'])}, "
                      f"cond p={pstr(aov['cond'])}, state×cond p={pstr(aov['inter'])}")
                ax.text(0.02, 0.98,
                        f"ANOVA  state {stars(aov['state']) or 'ns'} · "
                        f"cond {stars(aov['cond']) or 'ns'} · "
                        f"s×c {stars(aov['inter']) or 'ns'}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=legend_fs, color="0.15",
                        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

            ax.set_xticks(range(len(panel_states)))
            ax.set_xticklabels([st for st in panel_labels])
            ax.set(xlabel="behaviour state", ylabel=VAR_LABEL[var])
            if di == 0:
                ax.set_title(VAR_TITLE[var], fontsize=13, pad=18)
            if vi == 0:
                ax.text(-0.16, 0.5, drug, transform=ax.transAxes, rotation=90,
                        fontsize=15, fontweight="bold", va="center", ha="center")
            panel_letter(ax, chr(ord("C") + di * len(panel_vars) + vi), dx=-0.08)

    cond_handles = [Patch(facecolor=cond_colors[c], label=c) for c in conditions]
    fig.legend(cond_handles, conditions, ncol=3, fontsize=legend_fs, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), frameon=False)

    fig.savefig(f"{save_dir}/embed_composite_figure.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{save_dir}/embed_composite_figure.svg", bbox_inches="tight")
    print(f"Saved figure to {save_dir}/embed_composite_figure.png/.svg")


def main():
    sessions, state_speed, lag_of, model = get_labelled()
    make_figure(sessions, state_speed, lag_of, model, drugs)


if __name__ == "__main__":
    main()
