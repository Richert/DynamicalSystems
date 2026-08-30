"""Composite 3-row figure: AR-HMM behavioural states x SPN firing-rate dynamics.

Row 1  : example velocity trace of one session, shaded by HMM state.
Row 2  : (c1) per-state velocity & cumulative |acceleration| signatures;
         (c2) locomotor-state occupancy, % change vs Control (Amph + A+Drug x3);
         (c3) SPN dimensionality, % change vs Control (same conditions).
Row 3  : transition-triggered CENTER and HALF-WIDTH of the SPN firing-rate
         distribution (Control vs Amph vs A+Drug), one column per drug, same
         state transition (auto-selected for the clearest response).

Reuses analysis_arhmm_spn_states.py for loading/fitting/labelling; caches the
fitted HMM so the figure can be re-drawn without refitting.

  /home/rgast/conda/envs/arhmm/bin/python plotting_arhmm_figure.py
"""

import os
import pickle
import numpy as np
import analysis_arhmm_spn_states as A

save_dir = A.save_dir
# EXACT model produced by analysis_arhmm_spn_states.py; this script never refits.
model_path = f"{save_dir}/arhmm_model.pkl"
csv_path = f"{save_dir}/arhmm_spn_state_readouts.csv"
fs = A.fs
drugs = ["haloperidol"]
conditions = ["Control", "Amphetamine", "A + Drug"]
condition_labels = ["Control", "Amphetamine", f"A + {drugs[0].capitalize()}"]
locomotor_speed = 2.0        # states with mean speed above this = "locomotor"
example_condition = "Amphetamine"
example_win_s = 100.0         # length of the example window (row 1)
# states used for the bar+line panels (rows 2-3): rest and running/accelerating.
# bar panels: which HMM states to show as x-axis groups, and which 2 dependent
# variables to plot (one panel each). vars: occupancy | slope | center | dim
panel_states = [0, 2, 3]
panel_labels = ["rest", "slow", "fast"]
panel_vars = ["center", "inv_slope"]   # panel D = 1/slope (SPN rate heterogeneity)

# ---------------------------------------------------------------------------
# labelled sessions (fit once, cache)
# ---------------------------------------------------------------------------

def get_labelled():
    """Load the EXACT AR-HMM fitted by analysis_arhmm_spn_states.py (never refit
    here), then relabel + label the same sessions with it."""
    sessions = A.load_sessions()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run analysis_arhmm_spn_states.py first -- it fits "
            f"the AR-HMM and saves it here, and this figure reuses that exact fit.")
    m = pickle.load(open(model_path, "rb"))
    hmm, mu, sd = m["hmm"], m["mu"], m["sd"]
    if m.get("K") != A.K_states:
        print(f"NOTE: saved model K={m.get('K')} differs from current K_states={A.K_states}. "
              f"Using the saved model; re-run the analysis script to change K.")
    for s in sessions:
        s["obs_z"] = (s["obs"] - mu) / sd
    print(f"Loaded AR-HMM from analysis run (K={hmm.K}, lags={m.get('ar_lags')}, "
          f"kappa={m.get('kappa')}).")
    state_speed = A.relabel_by_speed(sessions, hmm, hmm.K)   # deterministic given hmm+sessions
    lag_of = A.animal_lags(sessions)
    return sessions, state_speed, lag_of

# ---------------------------------------------------------------------------
# transition helpers (for row 3 + selection)
# ---------------------------------------------------------------------------

def gather_transitions(sessions, frm, to):
    """List of (session, onset_frame) where state goes frm->to with stable dwell."""
    out = []
    W, dw = A.trans_halfwin, A.min_dwell
    for s in sessions:
        z = s["z"]
        chg = np.where(np.diff(z) != 0)[0] + 1
        for t in chg:
            if z[t - 1] != frm or z[t] != to:
                continue
            if t - dw < 0 or t + dw > len(z) or t - W < 0 or t + W + 1 > len(z):
                continue
            if np.all(z[t - dw:t] == frm) and np.all(z[t:t + dw] == to):
                out.append((s, t))
    return out

cdf_window_frames = 4        # +/- frames used for each neuron's mean rate (peri-transition CDF)

def center_halfwidth(instances, lag_of):
    """Peri-transition across-neuron firing-rate CDF vs time. At each offset the
    per-neuron rate is a mean over a small window (and over transition instances
    within a session), so the across-neuron distribution has real spread -- its
    centre (mean log10 rate) and half-width (~sigma; slope of the CDF = 1/sigma)
    are then well defined (a single-frame 'instantaneous' CDF is degenerate,
    since >84% of neurons are silent at any one frame -> zero half-width)."""
    from collections import defaultdict
    W, w = A.trans_halfwin, cdf_window_frames
    taus = np.arange(-W, W + 1)
    center = np.full(len(taus), np.nan)
    halfw = np.full(len(taus), np.nan)
    if not instances:
        return taus / fs, center, halfw
    by_sess = defaultdict(list)
    for s, t in instances:
        by_sess[id(s)].append((s, t))
    for j, tau in enumerate(taus):
        per_neuron = []
        for insts in by_sess.values():
            s = insts[0][0]
            lag = lag_of.get(s["mouse"], 0)
            acc = []
            for _, t0 in insts:
                f = t0 + tau - lag
                if f - w >= 0 and f + w + 1 <= s["rates"].shape[1]:
                    acc.append(s["rates"][:, f - w:f + w + 1].mean(axis=1))  # per-neuron windowed mean
            if acc:
                per_neuron.append(np.mean(acc, axis=0))                       # avg over instances
        if per_neuron:
            x = np.log10(np.clip(np.concatenate(per_neuron), *A.rate_lims))
            center[j] = np.mean(x)
            halfw[j] = 0.5 * (np.percentile(x, 84.1) - np.percentile(x, 15.9))
    return taus / fs, center, halfw

def choose_transition(sessions, K):
    """Pick the frm->to transition that (a) occurs in every drug x condition
    cell and (b) has the largest peri-transition center modulation (pooled)."""
    best, best_score = None, -np.inf
    for frm in range(K):
        for to in range(K):
            if frm == to:
                continue
            inst = gather_transitions(sessions, frm, to)
            if len(inst) < 40:
                continue
            # require presence in all drug x condition cells
            cells = {(s["drug"], s["condition"]) for s, _ in inst}
            if len(cells) < len(drugs) * len(conditions):
                continue
            _, c, _ = center_halfwidth(inst, {})   # lag ignored for scoring
            score = np.nanmax(c) - np.nanmin(c)
            if score > best_score:
                best, best_score = (frm, to), score
    return best

# ---------------------------------------------------------------------------
# row-2 summary metrics
# ---------------------------------------------------------------------------

COND5_ORDER = ["Control", "Amphetamine", "A+halo", "A+xano", "A+MP10"]
_ADRUG = {"haloperidol": "A+halo", "xanomeline": "A+xano", "MP10": "A+MP10"}

def cond5_of(condition, drug):
    """5-level condition label: Control / Amphetamine (pooled) + A+<drug>."""
    if condition == "Control":
        return "Control"
    if condition == "Amphetamine":
        return "Amphetamine"
    return _ADRUG.get(drug)            # "A + Drug" -> per drug

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

def agg_state_cond(df, col):
    """mean +/- sem across animals of per-(mouse, cond5, state) values of `col`."""
    pa = df.dropna(subset=[col]).groupby(["mouse", "cond5", "state"], as_index=False)[col].mean()
    return pa.groupby(["cond5", "state"])[col].agg(["mean", "sem"])

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def make_figure(sessions, state_speed, lag_of, rows_drugs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.stats import ttest_rel
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    import seaborn as sb

    # font sizes constrained to 12-16 pt (works in project_1.svg per Inkscape test):
    # ticks/legends/annotations at the 12 floor, panel letters at the 16 ceiling.
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    legend_fs = plt.rcParams["xtick.labelsize"]   # all legends match the tick labels
    panel_letter_fs = 16                           # 16 = the allowed max, like panel A's "A"
    cb = sb.color_palette("colorblind")
    cond_colors = {"Control": cb[0], "Amphetamine": cb[1], "A + Drug": cb[2]}
    cond_short = {"Control": "Ctrl", "Amphetamine": "Amph", "A + Drug": "A+D"}
    K = len(state_speed)
    state_colors = sb.color_palette("Set2", K)
    df = readouts_df(sessions, lag_of)

    # drug(s) shown as bar+line rows (works for 1 or more); keep only those present
    rows_drug = [d for d in rows_drugs if d in set(df["drug"])] or sorted(set(df["drug"]))[:1]
    n_rows = len(rows_drug)

    # width matches project_1.svg page width (989.28 pt) so the figure sits below
    # panel A at 1:1; two content rows -> panels B-E.
    fig_w_in = 989.27631 / 72.0
    fig = plt.figure(figsize=(fig_w_in, 3.7 + 3.5 * n_rows))
    gs = fig.add_gridspec(1 + n_rows, 4, height_ratios=[1.15] + [1.0] * n_rows,
                          hspace=0.55, wspace=0.55)

    def panel_letter(ax, letter, dx=-0.14):
        ax.text(dx, 1.05, letter, transform=ax.transAxes, fontsize=panel_letter_fs,
                fontweight="bold", va="bottom", ha="right")

    # ---- ROW 0, A (cols 0-2): example trace, 60 s ----
    ax = fig.add_subplot(gs[0, 0:3])
    Lwin = int(example_win_s * fs)
    target = list(panel_states)
    nsel = len(target)
    # search Control sessions for the window (of the requested duration) that
    # maximises the number of frames in the selected states while keeping their
    # distribution as uniform as possible: score = total_frames * (perplexity/nsel)
    best = None   # (score, session, i0, total, perplexity)
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
            H = -np.where(p > 0, p * np.log(p), 0.0).sum(0)     # entropy over selected states
        perplexity = np.exp(H)                                  # 1 (skewed) .. nsel (uniform)
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
    for k in panel_states:   # only shade the states selected for panels C-F
        ax.fill_between(t, 0, ymax, where=(z[i0:i1] == k), color=state_colors[k],
                        alpha=0.35, step="mid", linewidth=0)
    ax.set(xlim=(t[0], t[-1]), ylim=(0, ymax), xlabel="time (s)", ylabel="velocity (cm/s)")
    state_handles = [Patch(facecolor=state_colors[k], alpha=0.5,
                           label=panel_labels[i]) for i, k in enumerate(panel_states)]
    ax.legend(handles=state_handles, ncol=len(panel_labels), fontsize=legend_fs,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)   # where the title was
    panel_letter(ax, "B", dx=-0.05)

    # ---- ROW 0, B (col 3): per-state velocity & cumulative |accel| signatures ----
    ax = fig.add_subplot(gs[0, 3])
    # cumulative absolute acceleration: total |accel| integrated over time within
    # each state, per session, averaged across sessions (units cm/s = total |Δv|).
    nsess = len(sessions)
    cum_abs_acc = [np.sum([np.abs(s["obs"][s["z"] == k, 1]).sum() for s in sessions])
                   / (fs * nsess) for k in panel_states]
    sel_speed = [state_speed[k] for k in panel_states]
    sel_colors = [state_colors[k] for k in panel_states]
    xs = np.arange(len(panel_states))
    ax.bar(xs - 0.2, sel_speed, width=0.4, color=sel_colors)                            # velocity, filled
    axB2 = ax.twinx()
    axB2.bar(xs + 0.2, cum_abs_acc, width=0.4, facecolor="none", edgecolor=sel_colors, linewidth=1.6)  # cum |accel|, open
    ax.set(xlabel="HMM state", ylabel="velocity (cm/s)", xticks=xs)
    ax.set_xticklabels(panel_labels)
    axB2.set_ylabel("cumulative accel.", labelpad=2)
    ax.set_title("state signatures")
    ax.legend(handles=[Patch(facecolor="0.6", label="velocity"),
                       Patch(facecolor="none", edgecolor="0.35", label="acceleration")],
              fontsize=legend_fs, loc="upper left")
    panel_letter(ax, "C", dx=-0.3)

    # ---- ROWS 1-2: boxen panels (rows = drugs) with state x condition ANOVA ----
    def per_animal(drug, var):
        """Long df of per-(mouse, state, condition) values for one drug/variable."""
        d = df[(df["drug"] == drug) & (df["state"].isin(panel_states))
               & (df["condition"].isin(conditions))].dropna(subset=[var])
        return d.groupby(["mouse", "state", "condition"], as_index=False)[var].mean()

    def paired_p(pa, state, var, cA, cB):
        """Paired t-test across animals (same mice, two conditions within a state)."""
        w = pa[pa["state"] == state].pivot_table(index="mouse", columns="condition", values=var)
        if cA in w.columns and cB in w.columns:
            sub = w[[cA, cB]].dropna()
            if len(sub) >= 3 and np.ptp(sub[cA] - sub[cB]) > 0:
                return float(ttest_rel(sub[cA], sub[cB]).pvalue)
        return np.nan

    def two_way_anova(pa, var):
        """N(states) x 3(conditions) two-way ANOVA -> p(state), p(cond), p(interaction)."""
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
                 "inv_slope": "rate CDF 1/slope",
                 "center": "rate CDF center (log)", "dim": "SPN dimensionality"}
    VAR_TITLE = {"occupancy": "state occupancy", "slope": "drugs abd behavior modulate SPN rate heterogeneity",
                 "inv_slope": "drugs and behavior modulate SPN rate heterogeneity",
                 "center": "behavior modulates mean SPN rates", "dim": "SPN dimensionality"}
    VAR_ALIAS = {"dimensionality": "dim", "firing rate slope": "slope",
                 "firing rate center": "center", "firing_rate_slope": "slope"}
    pvars = [VAR_ALIAS.get(v, v) for v in panel_vars]
    ncond = len(conditions)
    box_w = 0.7
    off = box_w / ncond   # hue-dodge spacing used to place post-hoc stars

    for di, drug in enumerate(rows_drug):
        for vi, var in enumerate(pvars):
            ax = fig.add_subplot(gs[di + 1, 2 * vi:2 * vi + 2])
            pa = per_animal(drug, var)
            sb.boxenplot(data=pa, x="state", y=var, hue="condition", order=panel_states,
                         hue_order=conditions, palette=cond_colors, width=box_w,
                         linewidth=0.8, legend=False, ax=ax)
            # robust y-limits (1st-99th pct) so single extreme fliers don't blow up the axis,
            # with headroom on top for the ANOVA box + post-hoc stars
            vals = pa[var].to_numpy()
            lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
            rng0 = (hi - lo) or (abs(hi) or 1.0)
            ax.set_ylim(lo - 0.08 * rng0, hi + 0.35 * rng0)
            yr = ax.get_ylim()[1] - ax.get_ylim()[0]

            # post-hoc: per-state condition contrasts; highlight the most significant
            contrasts = []
            for si, st in enumerate(panel_states):
                for cA, cB in [("Control", "Amphetamine"), ("Amphetamine", "A + Drug")]:
                    p = paired_p(pa, st, var, cA, cB)
                    xc = si + ((conditions.index(cA) + conditions.index(cB)) / 2 - (ncond - 1) / 2) * off
                    contrasts.append(dict(p=p, xc=xc))
            minp = min((c["p"] for c in contrasts if np.isfinite(c["p"])), default=np.inf)
            star_y = ax.get_ylim()[1] - 0.16 * yr   # consistent row, just below the ANOVA box
            for c in contrasts:
                s = stars(c["p"])
                if not s:
                    continue
                hot = np.isfinite(c["p"]) and c["p"] == minp and c["p"] < 0.05
                ax.text(c["xc"], star_y, s, ha="center", va="center", zorder=6,
                        fontsize=13 if s != "n.s." else 8,
                        fontweight="bold" if s != "n.s." else "normal",
                        color=("crimson" if hot else ("0.1" if s != "n.s." else "0.55")))

            # two-way ANOVA annotation (also printed to console)
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
            ax.set(xlabel="HMM state", ylabel=VAR_LABEL[var])
            if di == 0:
                ax.set_title(VAR_TITLE[var], fontsize=14, pad=14)
            if vi == 0:
                ax.text(-0.26, 0.5, drug, transform=ax.transAxes, rotation=90,
                        fontsize=15, fontweight="bold", va="center", ha="center")
            panel_letter(ax, chr(ord("D") + di * len(panel_vars) + vi), dx=-0.08)

    # bottom condition legend (Control / Amphetamine / A + Drug)
    cond_handles = [Patch(facecolor=cond_colors[c], label=condition_labels[i]) for i, c in enumerate(conditions)]
    fig.legend(cond_handles, condition_labels, ncol=3, fontsize=legend_fs, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), frameon=False)

    fig.savefig(f"{save_dir}/arhmm_composite_figure.png", dpi=130, bbox_inches="tight")
    fig.savefig(f"{save_dir}/arhmm_composite_figure.svg", bbox_inches="tight")
    print(f"Saved figure to {save_dir}/arhmm_composite_figure.png/.svg")


def main():
    sessions, state_speed, lag_of = get_labelled()
    make_figure(sessions, state_speed, lag_of, drugs)


if __name__ == "__main__":
    main()
