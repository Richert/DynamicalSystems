"""Slide figures of the behavioural-state analysis (pose features + AR-HMM), built from the
panels of figures_pose_arhmm.py in the slide style (white background, Helvetica-like font
as the beamer theme, 16:9 text width ~6 in).

  behavior_features    keypoints and features on video frames + 60 s of feature traces
  behavior_hmm         state feature profiles + example speed trace with the state sequence
  behavior_states      one typical bout per state (egocentric keypoints)
  behavior_drugs       state composition of behaviour per drug condition

  conda run -n parker python figures_pose_slides.py
"""
import os
import numpy as np

import figures_pose_arhmm as F
from figures_pose_arhmm import A, plt, INK, INK2, MUTED, GRID, STATE_COLORS, STATE_LABELS

out_dir = os.environ.get("SLIDE_DIR", "/home/rgast/OneDrive/presentations/parker/images")
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42})

def save(fig, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.pdf")
    fig.savefig(f"{out_dir}/{name}.png", dpi=200)
    plt.close(fig)
    print(f"saved {out_dir}/{name}.pdf/.png")

def letters(fig, items, y):
    for x, l in items:
        fig.text(x, y, l, fontsize=10, fontweight="bold", color=INK)

def shown_states(states):
    """States shown on slides: all but the excluded rear / artefact state."""
    return [k for k in states.index if not states.excluded[k]]

# ---------------------------------------------------------------------------

def slide_features(states, sessions):
    ex = F.feature_example(states, sessions)
    fig = plt.figure(figsize=(6.0, 3.15))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.22, left=0.155, right=0.99,
                          top=0.92, bottom=0.11)
    top = gs[0].subgridspec(1, 4, wspace=0.08, width_ratios=[1.33, 1, 1, 1])
    for i, fn in enumerate((F.panel_frame, F.panel_body, F.panel_head, F.panel_movement)):
        ax = fig.add_subplot(top[i])
        if fn in (F.panel_body, F.panel_head):
            fn(ax, ex, title=None, values=False)
        else:
            fn(ax, ex, title=None)
        ax.set_title(["frame + keypoints", "body posture", "head posture",
                      "movement (0.4 s)"][i], loc="left", fontsize=8)
    rows = [("speed", "speed (cm/s)", "torso"), ("turn", "turn (°/s)", "torso"),
            ("head_angle", "head angle (°)", "head"), ("bend", "bend (°)", "bend"),
            ("body_len", "body length", "length")]
    axes = F.panel_feature_series(fig, gs[1], ex, rows=rows, window=60, marks=False,
                                  title=None, label_size=7, hspace=0.3)
    for ax in axes:
        ax.yaxis.set_major_locator(plt.MaxNLocator(2, prune="lower"))
        ax.tick_params(axis="y", labelsize=6)
    letters(fig, [(0.005, "a"), (0.005, "b")], 0.95)
    fig.texts[-1].set_position((0.005, 0.47))
    for t in fig.texts[-2:]:
        t.set_x(0.005)
    save(fig, "behavior_features")

def state_matrix_axes(fig, ax, M, show, names, title, cmap, vmin, vmax, ylabels, xlabel=None,
                      ylabel=None, fmt=None, thr=0.1, cbar=None):
    labels = [STATE_LABELS[names[k]] for k in show]
    F.heat(ax, M, labels, labels, cmap, vmin, vmax,
           fmt or (lambda v: f"{v:.2f}".lstrip("0")), thr)
    ax.set_xticks(range(M.shape[1]), labels if M.shape[1] == len(show) else ax.get_xticklabels(),
                  rotation=90)
    if ylabels is True:
        for l, k in zip(ax.get_yticklabels(), show):
            l.set_color(STATE_COLORS[names[k]])
            l.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    if M.shape[1] == len(show):
        for l, k in zip(ax.get_xticklabels(), show):
            l.set_color(STATE_COLORS[names[k]])
            l.set_fontweight("bold")
    ax.tick_params(labelsize=6.5)
    for t in ax.texts:
        t.set_fontsize(5)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, loc="left")
    if cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        cb = fig.colorbar(sm, cax=ax.inset_axes([1.04, 0.25, 0.05, 0.5]), ticks=cbar[0])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6)
        cb.set_label(cbar[1], color=INK2, labelpad=1)

def slide_hmm(states, sessions):
    """State feature profiles, state confusability (held-out decoder) and transitions
    (switches only), without the excluded rear / artefact state."""
    show = shown_states(states)
    names = states.name.tolist()
    K = len(states)
    fig = plt.figure(figsize=(6.0, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.5, left=0.17, right=0.985,
                          top=0.88, bottom=0.27)
    # (a) state profiles: mean z-scored feature per state
    ax = fig.add_subplot(gs[0])
    allX = np.concatenate([s["X"] for s in sessions])
    allz = np.concatenate([s["z"] for s in sessions])
    L = np.array([allX[allz == k].mean(0) for k in show])
    flab = [F.FEAT_LABELS[f].replace(" (log)", "") for f in A.features]
    F.heat(ax, L, [STATE_LABELS[names[k]] for k in show], flab, F.DIVERGING, -2, 2,
           lambda v: "", 9)
    for l, k in zip(ax.get_yticklabels(), show):
        l.set_color(STATE_COLORS[names[k]])
        l.set_fontweight("bold")
    ax.set_xticks(range(len(A.features)), flab, rotation=90)
    ax.tick_params(labelsize=6.5)
    ax.set_title("feature profiles", loc="left")
    sm = plt.cm.ScalarMappable(cmap=F.DIVERGING, norm=plt.Normalize(-2, 2))
    cb = fig.colorbar(sm, cax=ax.inset_axes([1.03, 0.3, 0.05, 0.4]), ticks=[-2, 0, 2])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("z", color=INK2, labelpad=1)
    # (b) confusability: held-out decoder of the state from the features (rows: HMM state)
    cm = F.decoder_confusion(sessions, K)[np.ix_(show, show)]
    cm = cm / cm.sum(1, keepdims=True)
    ax = fig.add_subplot(gs[1])
    state_matrix_axes(fig, ax, cm, show, names,
                      f"confusability (diag. {np.diag(cm).mean():.2f})", F.SEQUENTIAL, 0, 1,
                      False, xlabel="decoded state", ylabel="HMM state")
    # (c) transitions between states (switches only)
    T = np.zeros((K, K))
    for s in sessions:
        z = s["z"]
        sw = np.flatnonzero(np.diff(z) != 0)
        sw = sw[(z[sw] >= 0) & (z[sw + 1] >= 0)]                        # excluded frames: -1
        np.add.at(T, (z[sw], z[sw + 1]), 1)
    T = T[np.ix_(show, show)]
    T = T / T.sum(1, keepdims=True)
    T[np.eye(len(T), dtype=bool)] = np.nan
    ax = fig.add_subplot(gs[2])
    state_matrix_axes(fig, ax, T, show, names, "transitions (at a switch)", F.SEQUENTIAL, 0, 1,
                      False, xlabel="next state", ylabel="current state")
    letters(fig, [(0.005, "a"), (0.4, "b"), (0.705, "c")], 0.93)
    save(fig, "behavior_hmm")

def slide_arhmm(states, sessions):
    """How the AR-HMM works: graphical model + equations, and feature traces of the example
    session segmented by the inferred states."""
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
    from matplotlib.colors import ListedColormap
    names = states.name.tolist()
    s = next(x for x in sessions if x["session"] == F.example)
    t0w, t1w = F.example_window(s, states)
    # 20-s segment of the example window with the most state switches
    w = int(20 * A.fs)
    sw = np.r_[0, np.diff(s["z"]) != 0].cumsum()
    best = max(range(t0w, t1w - w, int(A.fs)), key=lambda a: sw[a + w] - sw[a])
    t0, t1 = best, best + w
    fig = plt.figure(figsize=(6.0, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.2, left=0.01, right=0.99,
                          top=0.9, bottom=0.14)
    # (a) graphical model
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    xs = [2.9, 4.95, 7.0, 9.05]
    zy, xy, r = 8.3, 5.4, 0.66
    labs = ["t−3", "t−2", "t−1", "t"]
    seq_names = [n for n in ("rest", "slow_walk", "slow_walk", "walk") if n in STATE_COLORS] \
        if "walk" in names else [names[0], names[2], names[2], names[3]]
    zcols = [STATE_COLORS[n] for n in seq_names]
    for x, l, c in zip(xs, labs, zcols):
        ax.add_patch(Circle((x, zy), r, fc=c, ec="white", lw=1.2, zorder=3))
        ax.text(x, zy, f"$z_{{{l}}}$", ha="center", va="center", color="white", fontsize=7.5,
                zorder=4)
        ax.add_patch(Circle((x, xy), r, fc="white", ec=INK2, lw=0.9, zorder=3))
        ax.text(x, xy, f"$x_{{{l}}}$", ha="center", va="center", color=INK, fontsize=7.5,
                zorder=4)
        ax.add_patch(FancyArrowPatch((x, zy - r), (x, xy + r), arrowstyle="-|>",
                                     mutation_scale=7, color=INK2, lw=0.8, zorder=2))
    for a, b in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((a + r, zy), (b - r, zy), arrowstyle="-|>",
                                     mutation_scale=7, color=INK2, lw=0.8))
    for a in xs[:-1]:                                     # AR: past observations -> x_t
        ax.add_patch(FancyArrowPatch((a + 0.3, xy - r + 0.1), (xs[-1] - 0.4, xy - r + 0.1),
                                     arrowstyle="-|>", mutation_scale=6, color=F.FC["head"],
                                     lw=0.8, connectionstyle=f"arc3,rad={0.12 + 0.1 * (xs[-1] - a) / 3}"))
    ax.text(0.0, zy, "hidden\nstate", fontsize=7, color=INK2, va="center")
    ax.text(0.0, xy, f"{len(A.features)} pose\nfeatures\n(20 Hz)", fontsize=7, color=INK2, va="center")
    ax.text(6.0, 2.45, "autoregressive: 4 lags (200 ms)", fontsize=6.5, color=F.FC["head"],
            ha="center")
    eq = (r"$x_t = \sum_{l=1}^{4} A_l^{(z_t)}\, x_{t-l} + b^{(z_t)} + \varepsilon_t,"
          r"\quad \varepsilon_t \sim \mathcal{N}(0, \Sigma^{(z_t)})$")
    ax.text(5.0, 1.25, eq, ha="center", va="center", fontsize=7.2, color=INK)
    ax.text(5.0, 0.15, r"$P(z_t = j \mid z_{t-1} = i)$: sticky ($+\kappa$ on self-transitions)"
            rf"  ·  $K = {A.final_K}$", ha="center", va="center", fontsize=6.6, color=INK2)
    ax.set_title("each state: its own linear feature dynamics", loc="left", fontsize=8)
    # (b) data: z-scored HMM inputs over 20 s, background = inferred state
    rows = [("log_speed", "speed"), ("turn_asinh", "turn"), ("head_angle", "head angle"),
            ("bend", "bend"), ("log_head_speed", "head speed")]
    sub = gs[1].subgridspec(len(rows) + 1, 1, height_ratios=[1] * len(rows) + [0.45],
                            hspace=0.12)
    t = np.arange(t1 - t0) / A.fs
    z = s["z"][t0:t1]
    st = np.flatnonzero(np.diff(np.r_[-1, z]) != 0)
    en = np.r_[st[1:], len(z)]
    X = s["X"]
    for i, (col, lab) in enumerate(rows):
        ax = fig.add_subplot(sub[i])
        for a, b in zip(st, en):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color=STATE_COLORS[names[z[a]]], alpha=0.18,
                       lw=0)
        ax.plot(t, X[t0:t1, A.features.index(col)], color=INK, lw=0.7)
        ax.set_xlim(0, t[-1])
        ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6.8, labelpad=2)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.set_title("example: features (z-scored), shaded by inferred state",
                         loc="left", fontsize=8)
    ax = fig.add_subplot(sub[-1])
    cmap = ListedColormap([STATE_COLORS[n] for n in names])
    ax.imshow(z[None], cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5, aspect="auto",
              extent=(0, t[-1], 0, 1), interpolation="nearest")
    ax.set_yticks([])
    ax.set_ylabel("state", rotation=0, ha="right", va="center", fontsize=6.8, labelpad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time (s)")
    # state names on the longest bouts
    for a, b in zip(st, en):
        if (b - a) / A.fs >= 1.2:
            ax.text((t[a] + t[min(b, len(t) - 1)]) / 2, 0.5, STATE_LABELS[names[z[a]]],
                    ha="center", va="center", fontsize=5.5, color="white")
    fig.subplots_adjust(left=0.01)
    letters(fig, [(0.005, "a"), (0.48, "b")], 0.93)
    save(fig, "behavior_arhmm")

def slide_states(states, sessions, b):
    # the paper figure's layout at slide size and font
    F.save, F.fig_dir = (lambda fig, name: save(fig, "behavior_states")), out_dir
    F.fig_states(states, sessions, b)

def slide_merge(states, sessions):
    """From fitted states to behaviour labels: mapping diagram (A.label_groups) and the speed
    distribution of each fitted state with the merged groups marked."""
    from matplotlib.patches import FancyBboxPatch
    names = states.name.tolist()
    lmap, labels = F.label_map(states)
    many = len(names) > 12                             # e.g. v7 (20 states): compact labels
    fs_box = 5.6 if many else 7
    fig = plt.figure(figsize=(6.0, 3.6 if many else 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.25, left=0.01, right=0.98,
                          top=0.88, bottom=0.15)
    # (a) mapping: fitted states (left) -> behaviour labels (right)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.invert_yaxis()
    ax.axis("off")
    ys = {k: k for k in range(len(names))}
    # label rows: centred on the mean row of their states
    ly = {li: np.mean([k for k in range(len(names)) if lmap[k] == li]) for li in range(len(labels))}
    # labels at least one row apart (a merged label's mean row can coincide with another's)
    order_l = sorted(ly, key=ly.get)
    for i in range(1, len(order_l)):
        ly[order_l[i]] = max(ly[order_l[i]], ly[order_l[i - 1]] + 1)
    shift = (max(ly.values()) - (len(names) - 1)) / 2
    if shift > 0:
        ly = {li: y - shift for li, y in ly.items()}
    for k, n in enumerate(names):
        li = lmap[k]
        merged = len(A.label_groups[labels[li]]) > 1
        ax.plot([3.7, 6.3], [ys[k], ly[li]], color=F.LABEL_COLORS[labels[li]] if merged else GRID,
                lw=1.6 if merged else 0.8, zorder=1)
        ax.add_patch(FancyBboxPatch((0.2, k - 0.36), 3.5, 0.72, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=STATE_COLORS[n], ec="white", lw=1, zorder=2))
        ax.text(1.95, k, n if many else STATE_LABELS[n], ha="center", va="center", fontsize=fs_box,
                color="white",
                fontweight="bold", zorder=3)
    for li, lab in enumerate(labels):
        merged = len(A.label_groups[lab]) > 1
        ax.add_patch(FancyBboxPatch((6.3, ly[li] - 0.36), 3.5, 0.72,
                                    boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=F.LABEL_COLORS[lab], ec=INK if merged else "white",
                                    lw=1.2 if merged else 1, zorder=2))
        ax.text(8.05, ly[li], F.LABEL_NAMES[lab], ha="center", va="center", fontsize=fs_box,
                color="white", fontweight="bold", zorder=3)
    ax.text(1.95, -0.85, "fitted HMM states", ha="center", fontsize=7.5, color=INK2)
    ax.text(8.05, -0.85, "behaviour labels", ha="center", fontsize=7.5, color=INK2)
    # (b) speed per fitted state (log), merged groups bracketed
    ax = fig.add_subplot(gs[1])
    allf = np.concatenate([s["feat"].speed.values for s in sessions])
    allz = np.concatenate([s["z"] for s in sessions])
    rng = np.random.default_rng(1)
    for k, n in enumerate(names):
        v = allf[allz == k]
        v = rng.choice(v, min(len(v), 20000), replace=False)
        q = np.percentile(v, [5, 25, 50, 75, 95])
        ax.plot([q[0], q[4]], [k, k], color=STATE_COLORS[n], lw=1)
        ax.plot([q[1], q[3]], [k, k], color=STATE_COLORS[n], lw=6, solid_capstyle="butt")
        ax.plot([q[2]], [k], "|", color="white", ms=6, mew=1.5)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 40)
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_yticks(range(len(names)), [n if many else STATE_LABELS[n] for n in names], fontsize=fs_box)
    for l, n in zip(ax.get_yticklabels(), names):
        l.set_color(STATE_COLORS[n])
        l.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, labelsize=7)
    ax.set_xlabel("torso speed (cm/s; box 25–75%, line 5–95%)")
    for lab, sts in A.label_groups.items():
        if len(sts) > 1:
            ks = [names.index(st) for st in sts]
            for kk in ks:                                  # member rows (groups can be non-contiguous)
                ax.axhspan(kk - 0.45, kk + 0.45, color=F.LABEL_COLORS[lab], alpha=0.12, lw=0)
            ax.text(38, np.mean(ks), f"→ {F.LABEL_NAMES[lab]}", ha="right", va="center",
                    fontsize=7, color=F.LABEL_COLORS[lab], fontweight="bold")
    ax.set_title("speed per fitted state", loc="left")
    letters(fig, [(0.005, "a"), (0.47, "b")], 0.93)
    save(fig, "behavior_merge")

def slide_drugs(states, sessions):
    show = shown_states(states)
    names = states.name.tolist()
    conds = getattr(A, "analysis_conditions", A.conditions) if "display" in A.cfg else A.conditions
    short = {"Control": "Control", f"{A.drug} low": "Cloz. low", f"{A.drug} high": "Cloz. high",
             "Amphetamine": "Amph.", f"A + {A.drug} low": "Amph. +\nCloz. low",
             f"A + {A.drug} high": "Amph. +\nCloz. high"}
    occ = np.array([[np.mean([np.mean(s["z"] == k) for s in sessions if s["condition"] == c])
                     for c in conds] for k in show])
    occ = occ / occ.sum(0, keepdims=True)                 # excluded state removed
    n = [sum(s["condition"] == c for s in sessions) for c in conds]
    fig, ax = plt.subplots(figsize=(6.0, 2.45))
    fig.subplots_adjust(left=0.08, right=0.78, top=0.85, bottom=0.17)
    x = np.array([0, 1, 2, 3.4, 4.4, 5.4]) if len(conds) == 6 else \
        np.r_[0, 1.4 + np.arange(len(conds) - 1)]          # gap after the drug-free control(s)
    bottom = np.zeros(len(conds))
    for i, k in enumerate(show):
        ax.bar(x, occ[i] * 100, 0.72, bottom=bottom * 100, color=STATE_COLORS[names[k]],
               edgecolor="white", lw=0.6, label=STATE_LABELS[names[k]])
        bottom += occ[i]
    ax.set_xticks(x, [short[c] for c in conds])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of time")
    ax.tick_params(axis="x", length=0)
    for xi, ni in zip(x, n):
        ax.text(xi, 101.5, f"n = {ni}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
              handlelength=1, fontsize=7)
    ax.set_title("time spent in each behavioural state (mean over sessions)", loc="left",
                 pad=12)
    save(fig, "behavior_drugs")


if __name__ == "__main__":
    states, sessions = F.load_data()
    slide_merge(states, sessions)                      # fitted states -> labels
    if "display" in A.cfg:                             # the rest on the merged labels
        states, sessions = F.label_level(states, sessions)
    slide_features(states, sessions)
    slide_arhmm(states, sessions)
    slide_hmm(states, sessions)
    slide_drugs(states, sessions)
    b = F.bouts(sessions)
    slide_states(states, sessions, b[b.state >= 0])
