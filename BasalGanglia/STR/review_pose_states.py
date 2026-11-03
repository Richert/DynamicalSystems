"""Review sheets of a fitted pose AR-HMM (for pruning / merging decisions), from the exported
state sequences in OUT_DIR (analysis_pose_arhmm.py export):

  review_summary.png   feature profiles (all features incl. wall distance / facing; values,
                       colour = z across states), occupancy per condition, median bout,
                       transitions (at a switch), confusability (held-out decoder)
  review_states_<i>.png  per state: speed / wall-distance / wall-facing distributions, three
                       example bouts (egocentric keypoints, start grey -> end dark) and their
                       torso paths in the arena (allocentric, arena circle from the video)

  OUT_DIR=.../pose_arhmm_v6 conda run -n parker python review_pose_states.py
"""
import os
import numpy as np
import pandas as pd

os.environ.setdefault("OUT_DIR", "/home/rgast/data/parker_data/pose_arhmm_v6")
import figures_pose_arhmm as F
from figures_pose_arhmm import A, plt

out = f"{A.out_dir}/review"
seed = 2
n_examples = 3
cols = ["speed", "turn", "abs_turn", "head_angle", "body_len", "bend", "head_speed", "abs_head_turn",
        "ear_width", "hip_width", "wall_dist", "wall_facing", "interp_share"]
labels = ["speed cm/s", "turn °/s (+ CCW)", "|turn| °/s", "head angle °", "body length", "bend °",
          "head speed cm/s", "|head turn| °/s", "ear width", "hip width", "wall dist cm",
          "wall facing (+ = wall)", "interpolated kp"]

def summary(states, sessions, b):
    K = len(states)
    names = [f"S{k}" for k in range(K)]
    allf = pd.concat([s["feat"] for s in sessions])
    allz = np.concatenate([s["z"] for s in sessions])
    prof = allf[cols].groupby(allz).mean().reindex(range(K))
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.35, wspace=0.3)
    ax = fig.add_subplot(gs[0, :2])
    z = ((prof - prof.mean()) / (prof.std() + 1e-12)).T.values
    ax.imshow(z, cmap=F.DIVERGING, vmin=-2, vmax=2, aspect="auto")
    for i in range(len(cols)):
        for j in range(K):
            v = prof.iloc[j, i]
            ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(cols)), labels, fontsize=8)
    ax.set_xticks(range(K), names, fontsize=8)
    ax.set_title("state profiles (mean; colour = z across states)", loc="left")
    # occupancy per condition + median bout
    ax = fig.add_subplot(gs[0, 2])
    conds = [c for c in A.conditions if any(s["condition"] == c for s in sessions)]
    occ = np.array([[np.mean([np.mean(s["z"] == k) for s in sessions if s["condition"] == c])
                     for c in conds] for k in range(K)])
    F.heat(ax, occ * 100, names, [c.replace("Clozapine", "Cloz.").replace("Amphetamine", "Amph.")
                                  for c in conds], F.SEQUENTIAL, 0, 40,
           lambda v: f"{v:.0f}" if v >= 1 else ("<1" if v > 0 else "0"), 0)
    med = b.groupby("state").dur.median().reindex(range(K))
    ax.set_yticks(range(K), [f"S{k}  ({med[k]:.2f} s)" for k in range(K)], fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_title("occupancy % per condition (median bout)", loc="left")
    # transitions
    ax = fig.add_subplot(gs[1, 0])
    T = np.zeros((K, K))
    for s in sessions:
        sw = np.flatnonzero(np.diff(s["z"]) != 0)
        np.add.at(T, (s["z"][sw], s["z"][sw + 1]), 1)
    T = T / np.maximum(T.sum(1, keepdims=True), 1)
    F.heat(ax, T, names, names, F.SEQUENTIAL, 0, 1, lambda v: f"{v:.2f}".lstrip("0"), 0.08,
           mask_diag=True)
    ax.set(xlabel="next state", ylabel="current state")
    ax.set_title("transitions (at a switch)", loc="left")
    # confusability
    ax = fig.add_subplot(gs[1, 1])
    cm = F.decoder_confusion(sessions, K)
    F.heat(ax, cm, names, names, F.SEQUENTIAL, 0, 1, lambda v: f"{v:.2f}".lstrip("0"), 0.08)
    ax.set(xlabel="decoded state", ylabel="HMM state")
    ax.set_title(f"confusability (held-out decoder, mean diag. {np.diag(cm).mean():.2f})", loc="left")
    ax = fig.add_subplot(gs[1, 2])
    for k in range(K):
        d = np.log10(b.dur[b.state == k].values)
        if len(d) > 2:
            vp = ax.violinplot(d, [k], widths=0.8, showextrema=False)
            for bd in vp["bodies"]:
                bd.set_facecolor("#2a78d6"); bd.set_alpha(0.6)
            ax.plot([k - 0.25, k + 0.25], [np.median(d)] * 2, color="k", lw=1.2)
    ax.axhline(np.log10(A.min_bout_s), color="#e34948", lw=0.8, ls="--")
    ax.set_yticks(np.log10([0.05, 0.1, 0.2, 0.5, 1, 3]), ["0.05", "0.1", "0.2", "0.5", "1", "3"])
    ax.set_xticks(range(K), names, fontsize=8)
    ax.set(ylabel="bout duration (s)")
    ax.set_title(f"bout durations (red: min. bout {A.min_bout_s} s)", loc="left")
    fig.suptitle(f"{os.path.basename(A.out_dir)}: K = {K}, kappa = {A.final_kappa:g} "
                 f"({len(sessions)} sessions)", fontsize=12)
    fig.savefig(f"{out}/review_summary.png", dpi=110, facecolor="white")
    plt.close(fig)

def example_bouts(b, k, rng):
    d = b[b.state == k]
    lo, hi = d.dur.quantile(0.5), d.dur.quantile(0.9)
    c = d[d.dur.between(lo, max(hi, lo + 0.05)) & (d.interp == 0)]
    if len(c) < n_examples:
        c = d.nsmallest(max(n_examples, 20), "interp")
    c = c.groupby("sess").head(1)                      # different sessions
    return c.sample(min(n_examples, len(c)), random_state=rng.integers(1 << 31))

def state_pages(states, sessions, b, per_page=5):
    K = len(states)
    rng = np.random.default_rng(seed)
    arena = pd.read_csv(A.arena_csv).set_index("session")
    allf = pd.concat([s["feat"] for s in sessions])
    allz = np.concatenate([s["z"] for s in sessions])
    cache = {}
    for page, k0 in enumerate(range(0, K, per_page)):
        ks = list(range(k0, min(K, k0 + per_page)))
        fig, axes = plt.subplots(len(ks), 3 + n_examples + 1, figsize=(22, 3.3 * len(ks)),
                                 gridspec_kw=dict(width_ratios=[1, 1, 1] + [0.8] * n_examples + [1.1]))
        axes = np.atleast_2d(axes)
        for row, k in zip(axes, ks):
            f = allf[allz == k]
            for ax, c, rng_, lab in ((row[0], "speed", (0, 25), "speed cm/s"),
                                     (row[1], "wall_dist", (0, 16), "wall distance cm"),
                                     (row[2], "wall_facing", (-1, 1), "wall facing")):
                ax.hist(allf[c].clip(*rng_), bins=40, range=rng_, density=True, color="#c3c2b7")
                ax.hist(f[c].clip(*rng_), bins=40, range=rng_, density=True, color="#2a78d6", alpha=0.8)
                ax.set_xlabel(lab, fontsize=8); ax.set_yticks([])
            occ = np.mean(allz == k)
            row[0].set_ylabel(f"S{k}\n{occ:.1%}", rotation=0, ha="right", va="center", fontsize=12,
                              fontweight="bold")
            ex = example_bouts(b, k, rng)
            arena_ax = row[-1]
            th = np.linspace(0, 2 * np.pi, 200)
            arena_ax.plot(np.cos(th), np.sin(th), color="#898781", lw=0.8)
            for e, (ax, (_, bt)) in enumerate(zip(row[3:3 + n_examples], ex.iterrows())):
                s = sessions[int(bt.sess)]["session"]
                if s not in cache:
                    cache[s] = F.keypoints(s)
                kp, _, sc, _, ix = cache[s]
                seq = kp[int(bt.start):int(bt.end) + 1]
                ego = F.egocentric(seq, ix)
                idx = np.unique(np.linspace(0, len(ego) - 1, min(len(ego), 6)).round().astype(int))
                for i, t in enumerate(idx):
                    w = i / max(len(idx) - 1, 1)
                    F.skeleton(ax, ego[t], ix, color=(1 - w) * np.array([0.83, 0.82, 0.8]) + w * np.array([0.05, 0.21, 0.42]),
                               lw=1.2 if t == idx[-1] else 0.8, ms=10 if t == idx[-1] else 4, edge="white", z=3 + i)
                ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-7, 10); ax.axis("off")
                ff = sessions[int(bt.sess)]["feat"].iloc[int(bt.start):int(bt.end)]
                ax.set_title(f"{s.split('_')[1]} {bt.dur:.2f} s, turn {ff.turn.sum() / A.fs:+.0f}°\n"
                             f"wall {ff.wall_dist.mean():.1f} cm", fontsize=7)
                # allocentric torso path (arena units: radius 1)
                key = [x for x in arena.index if x.endswith("/" + s)][0]
                a = arena.loc[key]
                px = seq / sc
                tors = px[:, [ix["neck"], ix["hip1"], ix["hip2"], ix["tail_base"]]].mean(1)
                p = (tors - [a.cx, a.cy]) / a.R_px
                col = plt.cm.tab10(e)
                arena_ax.plot(p[:, 0], -p[:, 1], color=col, lw=1.5)
                arena_ax.plot(p[-1, 0], -p[-1, 1], "o", color=col, ms=4)
            arena_ax.set_aspect("equal"); arena_ax.axis("off")
            arena_ax.set_title("torso paths in the arena", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out}/review_states_{page + 1}.png", dpi=90, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs(out, exist_ok=True)
    states, sessions = F.load_data()
    b = F.bouts(sessions)
    summary(states, sessions, b)
    state_pages(states, sessions, b)
    print(f"saved review sheets in {out}")
