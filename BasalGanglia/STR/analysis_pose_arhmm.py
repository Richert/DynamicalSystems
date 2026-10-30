"""Keypoint-based behavioural states: features from the MARS keypoints + sticky AR-HMM.

Per session (imaging period only, aligned with align_pose_to_imaging.py), from the 7
keypoints in cm (session cm/px scale), smoothed (Gaussian, smooth_s) against keypoint
jitter, at 20 Hz:
  log_speed   log10(torso speed + speed_eps), torso = mean of neck, hips, tail base
  turn        signed angular speed of the body heading (tail base -> neck), deg/s
  turn_asinh  arcsinh(turn / turn_eps): signed, log-like compression of the heavy tail
              (used for the HMM)
  head_angle  signed angle of the head (neck -> nose) relative to the body axis, deg
  body_len    nose - tail base length relative to the session median
  bend        signed body curvature: angle at the hip midpoint between the tail base ->
              hips and hips -> neck segments, deg (0 = straight; grooming, tight turns)
  head_speed  speed of the nose in the body frame (origin neck, x along the body axis),
              cm/s: head scanning / sniffing independent of locomotion
              (log_head_speed = log10(head_speed + speed_eps) for the HMM)
  head_turn   signed angular speed of the head relative to the body, deg/s
              (head_turn_asinh = arcsinh(head_turn / turn_eps) for the HMM)
  ear_width   ear - ear distance relative to the session median (shrinks in the top view
              when the head pitches up, e.g. rearing)
  hip_width   hip - hip distance relative to the session median (hunched vs stretched)
  wall_dist   torso distance to the arena wall (cm; stored, not used for the HMM)
Turn, head angle, bend and head turn are signed, so the HMM may form separate left /
right states.
Sign convention: positive = counter-clockwise as seen in the (top-down) video, i.e.
a left turn / head to the left if the video is not mirrored (image y axis points down).

Keypoints with a MARS confidence score < min_score are interpolated in time first
(interp_share = share of interpolated keypoints per frame, stored for QC).

The HMM features are z-scored globally (block means to fs_fit if below fs), and a
sticky AR-HMM (ssm) is fitted for each K in K_grid and stickiness kappa in kappa_grid
on a training split of sessions (stratified by condition, at most fit_cap_s per
session); K / kappa are judged by held-out log-likelihood per frame and state dwell
times. Outputs (pose_arhmm/): per-session features (20 Hz), the sweep (K_sweep.csv,
pose_arhmm_kappa_K_sweep.png), the fitted models and per kappa a figure with, per
candidate K, the state feature profiles, dwell times and occupancy per drug condition.

  /home/rgast/conda/envs/arhmm/bin/python analysis_pose_arhmm.py
Paths / processes can be set via POSE_ROOT, OFFSETS_CSV, OUT_DIR and N_PROCESSES (server).
"""

import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter1d

pose_root = os.environ.get("POSE_ROOT", "/home/rgast/data/parker_data/pose_data/Clozapine")
offsets_csv = os.environ.get("OFFSETS_CSV",
                             "/home/rgast/data/parker_data/pose_alignment/pose_imaging_offsets.csv")
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/pose_arhmm")
drug = "Clozapine"
fs = 20.0
arena_cm = 30.48
smooth_s = 0.1                     # Gaussian SD for keypoint smoothing (s)
min_score = 0.5                    # MARS keypoint confidence below which a keypoint is
                                   # interpolated (~5% of frames have one such keypoint)
speed_eps = 0.5                    # cm/s, offset of the log speed (rest threshold scale)
turn_eps = 5.0                     # deg/s, scale of the arcsinh compression of turning speed
features = ["log_speed", "turn_asinh", "head_angle", "body_len", "bend", "log_head_speed",
            "head_turn_asinh", "ear_width", "hip_width"]
conditions = ["Control", drug, "Amphetamine", f"A + {drug}"]
cond_of = {("Vehicle", "Control"): "Control", ("Vehicle", "Amph"): "Amphetamine",
           ("HighDose", "Control"): drug, ("HighDose", "Amph"): f"A + {drug}"}

# AR-HMM
fs_fit = 20.0                      # fit at the full video rate (no downsampling)
fit_cap_s = 900                    # at most 15 min per session in the fit
K_grid = [5, 6, 8]                 # K = 4 merged the turning / fast states (v3)
ar_lags = 4                        # 200 ms of context at 20 Hz
# stickiness: at kappa = 100 states switched every 50-300 ms (median dwell) at 20 Hz with
# 9 features; swept to reach realistic behavioural dwell times (~0.3-1 s). In ssm, kappa
# is a pseudo-count added to the diagonal of the expected transition counts (~1.1 M
# training frames, ~0.3 M state switches), so kappa = 1e3-1e5 hardly changed the median
# dwell (0.15 -> 0.2 s, v3); it has to be of the order of the per-state counts
kappa_grid = [3e5, 1e6, 3e6]
em_iters = 50
n_restarts = 3
heldout_frac = 0.2
n_jobs = int(os.environ.get("N_PROCESSES", 8))
seed = 1
show_K = [5, 6, 8]            # candidate K values summarised in the figures (one per kappa)

# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def angle_between(u, v):
    """Signed angle (deg) from vectors u to v, both (T, 2)."""
    a = np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], (u * v).sum(1))
    return np.degrees(a)

def session_features(f, off, n_frames, scale):
    with h5py.File(f, "r") as h:
        tr = h["tracks"][0]                                   # (2, 7, T)
        score = h["point_scores"][0].T                        # (T, 7)
        nodes = [n.decode() for n in h["node_names"][...]]
    kp = np.transpose(tr, (2, 1, 0)) * scale                  # (T, 7, 2) in cm
    # low-confidence keypoints (occlusion / tracking errors) -> linear interpolation in time
    low = score < min_score
    kp[low] = np.nan
    kp = pd.DataFrame(kp.reshape(len(kp), -1)).interpolate(limit_direction="both") \
        .values.reshape(kp.shape)
    interp_share = low[off:off + n_frames].mean(1)
    # arena centre from the keypoint extent over the whole video (as in the paper)
    lo = np.nanpercentile(kp.reshape(-1, 2), 0.01, axis=0)
    hi = np.nanpercentile(kp.reshape(-1, 2), 99.99, axis=0)
    centre = (lo + hi) / 2
    kp = kp[off:off + n_frames]
    kp = gaussian_filter1d(kp, sigma=smooth_s * fs, axis=0, mode="nearest")
    ix = {n: i for i, n in enumerate(nodes)}
    nose, neck, tail = kp[:, ix["nose"]], kp[:, ix["neck"]], kp[:, ix["tail_base"]]
    hips = (kp[:, ix["hip1"]] + kp[:, ix["hip2"]]) / 2
    torso = kp[:, [ix["neck"], ix["hip1"], ix["hip2"], ix["tail_base"]]].mean(1)
    speed = np.r_[0.0, np.linalg.norm(np.diff(torso, axis=0), axis=1)] * fs
    body = neck - tail
    heading = np.unwrap(np.arctan2(body[:, 1], body[:, 0]))
    # image y points down: negate so that positive = counter-clockwise on screen
    turn = gaussian_filter1d(-np.degrees(np.gradient(heading)) * fs, smooth_s * fs)
    head = -angle_between(body, nose - neck)
    head_turn = gaussian_filter1d(np.gradient(np.unwrap(np.radians(head))), smooth_s * fs)
    head_turn = np.degrees(head_turn) * fs
    bend = -angle_between(hips - tail, neck - hips)
    # nose in the body frame (origin neck, x along the body axis): head movement
    # relative to the body (scanning / sniffing), independent of locomotion
    c, s = np.cos(-heading), np.sin(-heading)
    rel = nose - neck
    ego = np.c_[c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]]
    head_speed = np.r_[0.0, np.linalg.norm(np.diff(ego, axis=0), axis=1)] * fs
    length = np.linalg.norm(nose - tail, axis=1)
    ears = np.linalg.norm(kp[:, ix["ear1"]] - kp[:, ix["ear2"]], axis=1)
    hipw = np.linalg.norm(kp[:, ix["hip1"]] - kp[:, ix["hip2"]], axis=1)
    return pd.DataFrame(dict(
        log_speed=np.log10(speed + speed_eps), speed=speed,
        turn=turn, head_angle=head, turn_asinh=np.arcsinh(turn / turn_eps),
        body_len=length / np.median(length), bend=bend,
        head_speed=head_speed, log_head_speed=np.log10(head_speed + speed_eps),
        head_turn=head_turn, head_turn_asinh=np.arcsinh(head_turn / turn_eps),
        ear_width=ears / np.median(ears), hip_width=hipw / np.median(hipw),
        wall_dist=arena_cm / 2 - np.linalg.norm(torso - centre, axis=1),
        abs_turn=np.abs(turn), abs_head_turn=np.abs(head_turn), interp_share=interp_share))

def load_all():
    offsets = pd.read_csv(offsets_csv).set_index("session")
    sessions = []
    for f in sorted(glob.glob(f"{pose_root}/*/*/*_pose_top_v1_8.h5")):
        dose, part = f.split("/")[-3:-1]
        base = os.path.basename(f).split("_pose_top")[0]
        key = f"{drug.lower()}/{dose}/{base}"
        if (dose, part) not in cond_of or key not in offsets.index:
            continue
        o = offsets.loc[key]
        feat = session_features(f, int(o.offset_frames), int(o.old_frames_5hz) * 4,
                                float(o.cm_per_px))
        sessions.append(dict(session=base, mouse=re.search(r"_(m\d+|f\d+)_", base).group(1),
                             condition=cond_of[(dose, part)], feat=feat))
    return sessions

# ---------------------------------------------------------------------------
# AR-HMM fits (process pool over K x restarts)
# ---------------------------------------------------------------------------

_DATA = {}

def _init(train, test):
    os.environ["OMP_NUM_THREADS"] = "1"
    _DATA["train"], _DATA["test"] = train, test

def _fit(task):
    import ssm
    import numpy.random as npr
    K, kappa, s = task
    npr.seed(s)
    hmm = ssm.HMM(K, len(features), observations="ar", transitions="sticky",
                  observation_kwargs=dict(lags=ar_lags), transition_kwargs=dict(kappa=kappa))
    hmm.fit(_DATA["train"], method="em", num_iters=em_iters, verbose=0)
    n_test = sum(len(x) for x in _DATA["test"])
    return K, kappa, float(hmm.log_likelihood(_DATA["train"])), \
        float(hmm.log_likelihood(_DATA["test"])) / n_test, hmm

def to_fit_rate(X):
    k = int(fs / fs_fit)
    n = len(X) // k * k
    return X[:n].reshape(-1, k, X.shape[1]).mean(1)

def main():
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    sessions = load_all()
    print(f"{len(sessions)} sessions, {sum(len(s['feat']) for s in sessions) / fs / 3600:.1f} h")
    for s in sessions:
        s["feat"].to_csv(f"{out_dir}/features_{s['session']}.csv.gz", index=False)
    allf = pd.concat([s["feat"][features] for s in sessions])
    mu, sd = allf.mean().values, allf.std().values

    # stratified held-out split of sessions
    test = set()
    for c in conditions:
        idx = [i for i, s in enumerate(sessions) if s["condition"] == c]
        test.update(rng.choice(idx, max(1, round(heldout_frac * len(idx))), replace=False).tolist())
    cap = int(fit_cap_s * fs_fit)
    def prep(s, crop):
        X = to_fit_rate((s["feat"][features].values - mu) / sd)
        if crop and len(X) > cap:
            t0 = rng.integers(0, len(X) - cap)
            X = X[t0:t0 + cap]
        return np.ascontiguousarray(X)
    train = [prep(s, True) for i, s in enumerate(sessions) if i not in test]
    held = [prep(s, True) for i, s in enumerate(sessions) if i in test]
    print(f"train {len(train)} / test {len(held)} sessions, "
          f"{sum(map(len, train)) / fs_fit / 3600:.1f} h of training data at {fs_fit:g} Hz")

    tasks = [(K, kap, int(rng.integers(0, 2 ** 31)))
             for kap in kappa_grid for K in K_grid for _ in range(n_restarts)]
    with ProcessPoolExecutor(n_jobs, initializer=_init, initargs=(train, held)) as ex:
        res = list(ex.map(_fit, tasks))
    best = {}
    for K, kap, ll_tr, ll_te, hmm in res:
        if (K, kap) not in best or ll_tr > best[(K, kap)][0]:
            best[(K, kap)] = (ll_tr, ll_te, hmm)
    sweep = pd.DataFrame([dict(K=K, kappa=kap, train_ll=v[0], heldout_ll_per_frame=v[1])
                          for (K, kap), v in sorted(best.items())])
    pickle.dump(dict(models={key: v[2] for key, v in best.items()}, mu=mu, sd=sd,
                     features=features, fs_fit=fs_fit,
                     test_sessions=[sessions[i]["session"] for i in sorted(test)]),
                open(f"{out_dir}/models_K_sweep.pkl", "wb"))
    dwell = []
    for kap in kappa_grid:
        models = {K: best[(K, kap)][2] for K in show_K if (K, kap) in best}
        dwell += summarise(sessions, models, mu, sd, sweep[sweep.kappa == kap], kap)
    sweep = sweep.merge(pd.DataFrame(dwell), on=["K", "kappa"], how="left")
    sweep.to_csv(f"{out_dir}/K_sweep.csv", index=False)
    print(sweep.round(4).to_string(index=False))
    sweep_figure(sweep)

def sweep_figure(sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    shades = ["#9ec5f0", "#2a78d6", "#0b3d7a"]
    for kap, c in zip(kappa_grid, shades):
        d = sweep[sweep.kappa == kap]
        axes[0].plot(d.K, d.heldout_ll_per_frame, "o-", color=c, label=f"kappa = {kap:g}")
        axes[1].plot(d.K, d.median_dwell_s, "o-", color=c, label=f"kappa = {kap:g}")
    axes[0].set(xlabel="K", ylabel="held-out log-likelihood / frame", title="held-out fit")
    axes[1].axhspan(0.3, 1.0, color="#e6e5e0", zorder=0)
    axes[1].set(xlabel="K", ylabel="median bout duration (s), all states",
                title="state dwell (grey: 0.3-1 s target range)", yscale="log")
    for a in axes:
        a.legend(fontsize=8, frameon=False)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.savefig(f"{out_dir}/pose_arhmm_kappa_K_sweep.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"saved {out_dir}/pose_arhmm_kappa_K_sweep.png")

# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def label(hmm, s, mu, sd):
    """Viterbi states at fs_fit for a whole session, renumbered later by speed."""
    X = np.ascontiguousarray(to_fit_rate((s["feat"][features].values - mu) / sd))
    return hmm.most_likely_states(X)

def summarise(sessions, models, mu, sd, sweep, kap):
    """Figure + occupancy table for one kappa; returns the pooled dwell statistics per K."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2 = "#1f1f1e", "#5f5e5a"
    colors = dict(zip(conditions, ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]))
    rows, dwell_stats = [], []
    fig, axes = plt.subplots(1 + len(models), 3, figsize=(18, 5.2 * (1 + len(models))),
                             layout="constrained", squeeze=False)
    ax = axes[0, 0]
    ax.plot(sweep.K, sweep.heldout_ll_per_frame, "o-", color=ink)
    ax.set(xlabel="K", ylabel="held-out log-likelihood / frame", title="K sweep (held-out sessions)")
    for a in axes[0, 1:]:
        a.axis("off")
    raw_cols = ["speed", "turn", "abs_turn", "head_angle", "body_len", "bend", "head_speed",
                "head_turn", "abs_head_turn", "ear_width", "hip_width", "wall_dist", "interp_share"]
    raw_labels = ["speed cm/s", "turn deg/s (+ = CCW)", "|turn| deg/s", "head deg (+ = CCW)",
                  "body len (rel)", "bend deg (+ = CCW)", "head speed cm/s",
                  "head turn deg/s (+ = CCW)", "|head turn| deg/s", "ear width (rel)",
                  "hip width (rel)", "wall dist cm", "interpolated keypoints"]
    for r, (K, hmm) in enumerate(models.items(), start=1):
        lab = {s["session"]: label(hmm, s, mu, sd) for s in sessions}
        # per-state raw feature means (20 Hz features, labels upsampled)
        allz = np.concatenate([np.repeat(lab[s["session"]], int(fs / fs_fit))[:len(s["feat"])]
                               for s in sessions])
        allf = pd.concat([s["feat"].iloc[:len(np.repeat(lab[s["session"]], int(fs / fs_fit)))]
                          for s in sessions])
        prof = allf[raw_cols].groupby(allz[:len(allf)]).mean()
        order = prof.speed.sort_values().index.tolist()          # states by speed
        prof = prof.loc[order]
        remap = {old: new for new, old in enumerate(order)}
        # occupancy + dwell per condition
        occ, dwell = {}, {c: [] for c in conditions}
        for s in sessions:
            z = np.vectorize(remap.get)(lab[s["session"]])
            occ.setdefault(s["condition"], []).append(np.bincount(z, minlength=K) / len(z))
            st = np.flatnonzero(np.diff(np.r_[-1, z]) != 0)
            en = np.r_[st[1:], len(z)]
            dwell[s["condition"]] += list(zip(z[st], (en - st) / fs_fit))
            for k in range(K):
                rows.append(dict(K=K, kappa=kap, state=k, session=s["session"], mouse=s["mouse"],
                                 condition=s["condition"], occupancy=np.mean(z == k)))
        durs = np.array([d for c in conditions for _, d in dwell[c]])
        dwell_stats.append(dict(K=K, kappa=kap, median_dwell_s=float(np.median(durs)),
                                mean_dwell_s=float(durs.mean())))
        # (a) state profiles (z-scored across states, raw means annotated)
        ax = axes[r, 0]
        zprof = (prof - prof.mean()) / (prof.std() + 1e-12)
        ax.imshow(zprof.T.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        for i in range(K):
            for j, c in enumerate(raw_cols):
                ax.text(i, j, f"{prof.iloc[i][c]:.2g}", ha="center", va="center", fontsize=7)
        ax.set_yticks(range(len(raw_cols)), raw_labels, fontsize=8)
        ax.set_xticks(range(K), [f"S{k}" for k in range(K)], fontsize=8)
        ax.set_title(f"K = {K}: state profiles (mean feature values; colour = z across states)",
                     fontsize=9, loc="left")
        # (b) occupancy per condition
        ax = axes[r, 1]
        w = 0.8 / len(conditions)
        for i, c in enumerate(conditions):
            o = np.array(occ.get(c, []))
            ax.bar(np.arange(K) + (i - 1.5) * w, o.mean(0), w, yerr=o.std(0) / np.sqrt(len(o)),
                   color=colors[c], label=c, capsize=1.5, error_kw=dict(lw=0.6))
        ax.set_xticks(range(K), [f"S{k}" for k in range(K)], fontsize=8)
        ax.set(ylabel="occupancy", title=f"K = {K}: occupancy per condition (mean ± SEM)")
        ax.title.set_fontsize(9)
        ax.legend(fontsize=7, frameon=False)
        # (c) median dwell time per state and condition
        ax = axes[r, 2]
        for i, c in enumerate(conditions):
            dd = pd.DataFrame(dwell[c], columns=["state", "dur"])
            med = dd.groupby("state").dur.median().reindex(range(K))
            ax.plot(range(K), med.values, "o-", color=colors[c], label=c, ms=4)
        ax.set_yscale("log")
        ax.set_xticks(range(K), [f"S{k}" for k in range(K)], fontsize=8)
        ax.set(ylabel="median dwell (s)", title=f"K = {K}: median state dwell time")
        ax.title.set_fontsize(9)
        for a in axes[r]:
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    fig.suptitle(f"Keypoint-feature AR-HMM, kappa = {kap:g}: K sweep and candidate state sets "
                 f"(states ordered by speed)", fontsize=11, color=ink)
    out = f"{out_dir}/pose_arhmm_K_candidates_kappa{kap:g}.png"
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(f"{out_dir}/state_occupancy_candidates_kappa{kap:g}.csv", index=False)
    print(f"saved {out}")
    return dwell_stats


if __name__ == "__main__":
    main()
