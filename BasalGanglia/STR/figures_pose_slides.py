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

def slide_hmm(states, sessions):
    show = shown_states(states)
    names = states.name.tolist()
    labels = [STATE_LABELS[names[k]] for k in show]
    fig = plt.figure(figsize=(6.0, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.6], wspace=0.42, left=0.14, right=0.99,
                          top=0.88, bottom=0.3)
    # (a) state profiles: mean z-scored feature per state
    ax = fig.add_subplot(gs[0])
    allX = np.concatenate([s["X"] for s in sessions])
    allz = np.concatenate([s["z"] for s in sessions])
    L = np.array([allX[allz == k].mean(0) for k in show])
    flab = [F.FEAT_LABELS[f].replace(" (log)", "") for f in A.features]
    F.heat(ax, L, labels, flab, F.DIVERGING, -2, 2, lambda v: "", 9)
    for l, k in zip(ax.get_yticklabels(), show):
        l.set_color(STATE_COLORS[names[k]])
        l.set_fontweight("bold")
    ax.set_xticks(range(len(A.features)), flab, rotation=90)
    ax.tick_params(labelsize=6.5)
    ax.set_title("state feature profiles", loc="left")
    sm = plt.cm.ScalarMappable(cmap=F.DIVERGING, norm=plt.Normalize(-2, 2))
    cb = fig.colorbar(sm, cax=ax.inset_axes([1.04, 0.25, 0.05, 0.5]), ticks=[-2, 0, 2])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("z", color=INK2, labelpad=1)
    # (b) example: speed and state sequence (first 60 s of the example window)
    s = next(x for x in sessions if x["session"] == F.example)
    t0, _ = F.example_window(s, states)
    t1 = t0 + int(60 * A.fs)
    sub = gs[1].subgridspec(2, 1, height_ratios=[1, 0.2], hspace=0.06)
    t = np.arange(t1 - t0) / A.fs
    ax = fig.add_subplot(sub[0])
    ax.plot(t, s["feat"].speed.values[t0:t1], color=INK2, lw=0.7)
    ax.set_xlim(0, 60)
    ax.set_ylabel("speed (cm/s)")
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("example: speed and AR-HMM state sequence", loc="left")
    ax2 = fig.add_subplot(sub[1], sharex=ax)
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap([STATE_COLORS[n] for n in names])
    ax2.imshow(s["z"][t0:t1][None], cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5, aspect="auto",
               extent=(0, 60, 0, 1), interpolation="nearest")
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("time (s)")
    fig.legend(handles=[Patch(color=STATE_COLORS[names[k]], label=STATE_LABELS[names[k]])
                        for k in show], ncol=5, loc="lower left", bbox_to_anchor=(0.43, 0.0),
               frameon=False, handlelength=1, columnspacing=0.9, fontsize=6.8)
    letters(fig, [(0.005, "a"), (0.39, "b")], 0.93)
    save(fig, "behavior_hmm")

def slide_states(states, sessions, b):
    # the paper figure's layout at slide size and font
    F.save, F.fig_dir = (lambda fig, name: save(fig, "behavior_states")), out_dir
    F.fig_states(states, sessions, b)

def slide_drugs(states, sessions):
    show = shown_states(states)
    names = states.name.tolist()
    conds = A.conditions
    short = {"Control": "Control", f"{A.drug} low": "Cloz. low", f"{A.drug} high": "Cloz. high",
             "Amphetamine": "Amph.", f"A + {A.drug} low": "Amph. +\nCloz. low",
             f"A + {A.drug} high": "Amph. +\nCloz. high"}
    occ = np.array([[np.mean([np.mean(s["z"] == k) for s in sessions if s["condition"] == c])
                     for c in conds] for k in show])
    occ = occ / occ.sum(0, keepdims=True)                 # excluded state removed
    n = [sum(s["condition"] == c for s in sessions) for c in conds]
    fig, ax = plt.subplots(figsize=(6.0, 2.45))
    fig.subplots_adjust(left=0.08, right=0.78, top=0.85, bottom=0.17)
    x = np.array([0, 1, 2, 3.4, 4.4, 5.4])               # gap between drug-only and amph
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
    slide_features(states, sessions)
    slide_hmm(states, sessions)
    slide_drugs(states, sessions)
    slide_states(states, sessions, F.bouts(sessions))
