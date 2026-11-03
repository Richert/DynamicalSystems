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
  wall_dist   torso distance to the arena wall (cm); arena circle from the top-view video
              (arena_circles.py: Hough fit; LowDose without video: same mouse's nearest
              video day), which also sets the cm/px scale (arena diameter arena_cm).
              (wall_dist_c = clipped to [0, wall_clip_cm] for the HMM, v6: beyond a few cm
              the distance is location, not behaviour)
  wall_ahead  distance (cm) from the torso to the arena wall along the body heading (tail base
              -> neck): from wall_dist (facing the nearest wall) up to the arena diameter; v7
              HMM input together with the raw wall_dist
  wall_facing cos of the angle between the body heading and the outward radial direction
              (1 = facing the wall, 0 = parallel, -1 = facing the centre), weighted by
              proximity max(0, 1 - wall_dist / wall_clip_cm) (undefined far from the wall);
              v6: separates moving / turning along the wall from open-field running
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
  /home/rgast/conda/envs/arhmm/bin/python analysis_pose_arhmm.py export   (selected model ->
      per-session state sequences, see export_states)
  /home/rgast/conda/envs/arhmm/bin/python analysis_pose_arhmm.py summary  (per-session
      distance travelled / immobile share -> session_behavior.csv)
Paths / processes can be set via POSE_ROOT, OFFSETS_CSV, ARENA_CSV, OUT_DIR and N_PROCESSES
(server).
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
arena_csv = os.environ.get("ARENA_CSV", os.path.join(os.path.dirname(offsets_csv),
                                                     "arena_circles.csv"))
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/pose_arhmm")
drug = "Clozapine"
fs = 20.0
arena_cm = 30.48
smooth_s = 0.1                     # Gaussian SD for keypoint smoothing (s)
min_score = 0.5                    # MARS keypoint confidence below which a keypoint is
                                   # interpolated (~5% of frames have one such keypoint)
speed_eps = 0.5                    # cm/s, offset of the log speed (rest threshold scale)
turn_eps = 5.0                     # deg/s, scale of the arcsinh compression of turning speed
wall_clip_cm = 8.0                 # wall features: distance clipped / proximity weight scale
immobile_cm_s = 1.0                # session summary: immobile below this torso speed
# HMM inputs (v6: + wall distance and wall facing; the v5 model used the first 9, the fitted
# models store their feature list)
# v7: raw wall distance (minimum, cm) and distance to the wall ahead along the heading (cm);
# v6 used the clipped distance and the proximity-weighted facing, which were constant beyond
# 8 cm and produced a pure location state (all frames > 8 cm from the wall)
features = ["log_speed", "turn_asinh", "head_angle", "body_len", "bend", "log_head_speed",
            "head_turn_asinh", "ear_width", "hip_width", "wall_dist", "wall_ahead"]
conditions = ["Control", f"{drug} low", f"{drug} high", "Amphetamine", f"A + {drug} low",
              f"A + {drug} high"]
cond_of = {("Vehicle", "Control"): "Control", ("Vehicle", "Amph"): "Amphetamine",
           ("LowDose", "Control"): f"{drug} low", ("LowDose", "Amph"): f"A + {drug} low",
           ("HighDose", "Control"): f"{drug} high", ("HighDose", "Amph"): f"A + {drug} high"}
# the model was fitted on Vehicle / HighDose (v5); LowDose (added later) is only labelled
# with it (export), so that the states stay identical across doses
fit_conditions = ["Control", f"{drug} high", "Amphetamine", f"A + {drug} high"]

# AR-HMM
fs_fit = 20.0                      # fit at the full video rate (no downsampling)
fit_cap_s = 900                    # at most 15 min per session in the fit
# K = 4 merged the turning / fast states (v3); K = 5/6 merged the rare rearing-like state
# with clockwise turning (v4)
# v6 (wall features): K 10 / 12 / 15 at kappa 1e7 (kappa scaled up with K, as it acts on
# the per-state transition counts), larger K to prune / merge states afterwards
K_grid = [20]                      # v7: large K, pruned / merged after review
ar_lags = 4                        # 200 ms of context at 20 Hz
# stickiness: at kappa = 100 states switched every 50-300 ms (median dwell) at 20 Hz with
# 9 features; swept to reach realistic behavioural dwell times (~0.3-1 s). In ssm, kappa
# is a pseudo-count added to the diagonal of the expected transition counts (~1.1 M
# training frames, ~0.3 M state switches), so kappa = 1e3-1e5 hardly changed the median
# dwell (0.15 -> 0.2 s, v3); it has to be of the order of the per-state counts.
# v4: 3e5 -> 3e6 raised the median dwell 0.2 -> 0.3 s (K = 8) at a held-out cost of only
# 0.04 nats/frame
# v5: 6e6 / 9e6 gave the same median dwell (0.3 s) at K = 10
kappa_grid = [1e7, 2e7]
em_iters = 50
n_restarts = 10                    # 20 fits: one round on ~70 server workers
heldout_frac = 0.2
n_jobs = int(os.environ.get("N_PROCESSES", 8))
seed = 1
show_K = [20]              # candidate K values summarised in the figures (one per kappa)

# selected model (v5): K = 10 / kappa = 6e6; K = 8 merged fast turning with the rearing /
# artefact state, K = 12 only added near-empty or ~0.2 s states (held-out gain within the
# restart noise). States are numbered by mean speed (as in the figures).
# Turning states are not direction-specific per bout despite their mean turn: turn_fast
# (mean -18 deg/s) has CW and CCW bouts (net turn > 20 deg: 39% CW / 33% CCW under
# amphetamine), reorient (mean +17 deg/s, 0.2 s bouts) turns < 20 deg net in ~90% of bouts
# (brief head-led jerks, 58% followed by turn_fast). Direction has to come from the signed
# turn within the bout.
# per model version (out_dir basename): selected K / kappa, state names (numbered by mean speed),
# excluded states and the label table (label -> fitted states, display order)
model_cfg = {
    "pose_arhmm_v5": dict(
        K=10, kappa=6e6,
        names=["rest", "quiet", "shift", "slow_walk", "reorient", "walk", "rear_artifact",
               "turn_fast", "run", "run_fast"],
        # rearing / tracking artefact (5-7% interpolated keypoints)
        excluded=["rear_artifact"],
        # rest + quiet: both immobile (median speed 0.04 / 0.07 cm/s), most switches between
        # the two, decoder confusion 0.34; run + run_fast: the same locomotion at two speeds
        labels={"rest": ["rest", "quiet"], "shift": ["shift"], "slow_walk": ["slow_walk"],
                "walk": ["walk"], "run": ["run", "run_fast"], "reorient": ["reorient"],
                "turn_fast": ["turn_fast"], "rear_artifact": ["rear_artifact"]}),
    # v6 (+ wall distance / facing): K = 15 / kappa = 1e7, to be pruned / merged after review;
    # provisional names s00-s14 (one label per state) until then
    "pose_arhmm_v6": dict(
        K=15, kappa=1e7, names=[f"s{i:02d}" for i in range(15)], excluded=[],
        labels={f"s{i:02d}": [f"s{i:02d}"] for i in range(15)}),
    # v7 (raw wall distance + wall ahead): K = 20, kappa chosen from 1e7 / 2e7 after review
    # selected: kappa 1e7 (held-out 37.70 vs 37.60 at 2e7, no systematically longer bouts).
    # Review 1 (user, 2026-11-03): s05 / s06 / s10 excluded (rearing / tracking artefact: body
    # length 0.56-0.76, 7% interpolated keypoints, < 1%); merged s01 + s02, s03 + s04.
    # Review 2: merged rest + quiet / shift (s00-s02: rest -> quiet 98% of switches, decoded as
    # each other 38%), walk A + B (s07 + s08, confusion 0.33-0.49), sharp turns A + B (s13 + s14,
    # both fast clockwise turns). Label names provisional.
    "pose_arhmm_v7": dict(
        K=20, kappa=1e7, names=[f"s{i:02d}" for i in range(20)],
        excluded=["s05", "s06", "s10"],
        labels={"rest": ["s00", "s01", "s02"], "slow_walk": ["s03", "s04"],
                "walk": ["s07", "s08"], "scan_walk": ["s09"], "walk_ccw": ["s11"],
                "walk_fast": ["s12"], "turn_fast": ["s13", "s14"],
                "run_turn_ccw": ["s15"], "run": ["s16"], "run_turn_cw": ["s17"],
                "run_wall": ["s18"], "run_open": ["s19"], "rear_artifact": ["s05", "s06", "s10"]},
        display={"rest": ("rest", "#0d366b"), "slow_walk": ("slow walk", "#86b6ef"),
                 "walk": ("walk", "#ef8a55"), "scan_walk": ("scanning walk", "#d9a400"),
                 "walk_ccw": ("walk (CCW)", "#eb6834"), "walk_fast": ("fast walk", "#c4511f"),
                 "turn_fast": ("sharp turn (CW)", "#5140c4"),
                 "run_turn_ccw": ("turning run (CCW)", "#e87ba4"), "run": ("run", "#8f3a14"),
                 "run_turn_cw": ("turning run (CW)", "#b0306a"), "run_wall": ("wall run", "#5a1f0c"),
                 "run_open": ("open-field run", "#008300"),
                 "rear_artifact": ("rear / artefact", "#1baf7a")}),
    # same fit, kappa 2e7 (exported into its own directory; models / features linked)
    "pose_arhmm_v7_k2e7": dict(
        K=20, kappa=2e7, names=[f"s{i:02d}" for i in range(20)], excluded=[],
        labels={f"s{i:02d}": [f"s{i:02d}"] for i in range(20)}),
}
cfg = model_cfg.get(os.path.basename(os.path.normpath(out_dir)), model_cfg["pose_arhmm_v5"])
final_K, final_kappa = cfg["K"], cfg["kappa"]
state_names, excluded_states, label_groups = cfg["names"], cfg["excluded"], cfg["labels"]
# export: bouts shorter than min_bout_s are unlabelled in state_clean (-1): clean states over
# coverage for the behaviour-conditioned analyses
min_bout_s = 0.2
label_of = {st: lab for lab, sts in label_groups.items() for st in sts}

# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def angle_between(u, v):
    """Signed angle (deg) from vectors u to v, both (T, 2)."""
    a = np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], (u * v).sum(1))
    return np.degrees(a)

def arena(key):
    """(cm per px, arena centre in px) of a session (key drug/dose/session) from the arena
    table (arena_circles.py)."""
    a = pd.read_csv(arena_csv).set_index("session").loc[key]
    return arena_cm / 2 / float(a.R_px), np.array([a.cx, a.cy], float)

def load_keypoints(f, off, n_frames, scale, centre_px=None):
    """Keypoints of the imaging period in cm (T, 7, 2): low-confidence keypoints
    interpolated, smoothed; with the per-frame interpolated share, the arena centre (cm;
    centre_px given: from the arena table, else the keypoint extent) and the node names."""
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
    if centre_px is not None:
        centre = np.asarray(centre_px, float) * scale
    else:                                         # keypoint extent (fails with little exploration)
        lo = np.nanpercentile(kp.reshape(-1, 2), 0.01, axis=0)
        hi = np.nanpercentile(kp.reshape(-1, 2), 99.99, axis=0)
        centre = (lo + hi) / 2
    kp = kp[off:off + n_frames]
    kp = gaussian_filter1d(kp, sigma=smooth_s * fs, axis=0, mode="nearest")
    return kp, interp_share, centre, nodes

def session_features(f, off, n_frames, scale, centre_px=None):
    kp, interp_share, centre, nodes = load_keypoints(f, off, n_frames, scale, centre_px)
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
    radial = torso - centre
    wall_dist = arena_cm / 2 - np.linalg.norm(radial, axis=1)
    out = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-6)
    facing = (body * out).sum(1) / np.maximum(np.linalg.norm(body, axis=1), 1e-6)
    prox = np.clip(1 - wall_dist / wall_clip_cm, 0, 1)
    # distance from the torso to the arena wall along the body heading (ray-circle intersection)
    hu = body / np.maximum(np.linalg.norm(body, axis=1, keepdims=True), 1e-6)
    pu = (radial * hu).sum(1)
    R = arena_cm / 2
    wall_ahead = -pu + np.sqrt(np.maximum(pu ** 2 - (radial ** 2).sum(1) + R ** 2, 0))
    return pd.DataFrame(dict(
        log_speed=np.log10(speed + speed_eps), speed=speed,
        turn=turn, head_angle=head, turn_asinh=np.arcsinh(turn / turn_eps),
        body_len=length / np.median(length), bend=bend,
        head_speed=head_speed, log_head_speed=np.log10(head_speed + speed_eps),
        head_turn=head_turn, head_turn_asinh=np.arcsinh(head_turn / turn_eps),
        ear_width=ears / np.median(ears), hip_width=hipw / np.median(hipw),
        wall_dist=wall_dist, wall_dist_c=np.clip(wall_dist, 0, wall_clip_cm),
        wall_facing=facing * prox, wall_ahead=wall_ahead,
        abs_turn=np.abs(turn), abs_head_turn=np.abs(head_turn), interp_share=interp_share))

def load_all(only=None):
    """Features of every session with a known condition and pose-imaging offset
    (only: restrict to these session names)."""
    offsets = pd.read_csv(offsets_csv).set_index("session")
    sessions = []
    for f in sorted(glob.glob(f"{pose_root}/*/*/*_pose_top_v1_8.h5")):
        dose, part = f.split("/")[-3:-1]
        base = os.path.basename(f).split("_pose_top")[0]
        key = f"{drug.lower()}/{dose}/{base}"
        if (dose, part) not in cond_of or key not in offsets.index or \
                (only is not None and base not in only):
            continue
        o = offsets.loc[key]
        scale, centre_px = arena(key)
        feat = session_features(f, int(o.offset_frames), int(o.old_frames_5hz) * 4,
                                scale, centre_px)
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
    sessions = [s for s in load_all() if s["condition"] in fit_conditions]
    print(f"{len(sessions)} sessions, {sum(len(s['feat']) for s in sessions) / fs / 3600:.1f} h")
    for s in sessions:
        s["feat"].to_csv(f"{out_dir}/features_{s['session']}.csv.gz", index=False)
    allf = pd.concat([s["feat"][features] for s in sessions])
    mu, sd = allf.mean().values, allf.std().values

    # stratified held-out split of sessions
    test = set()
    for c in fit_conditions:
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

def label(hmm, s, mu, sd, feats=None):
    """Viterbi states at fs_fit for a whole session, renumbered later by speed (feats: the
    model's input features, default the current ones)."""
    X = np.ascontiguousarray(to_fit_rate((s["feat"][feats or features].values - mu) / sd))
    return hmm.most_likely_states(X)

def summarise(sessions, models, mu, sd, sweep, kap):
    """Figure + occupancy table for one kappa; returns the pooled dwell statistics per K."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2 = "#1f1f1e", "#5f5e5a"
    conditions = fit_conditions
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

# ---------------------------------------------------------------------------
# export of the selected model
# ---------------------------------------------------------------------------

def export_states():
    """Viterbi states of the selected model (final_K, final_kappa) for every session, from
    the features and models_K_sweep.pkl in out_dir (features of sessions added later, e.g.
    LowDose, are computed and stored first): states_<session>.csv.gz (20 Hz, imaging period,
    numbered by mean speed in the fitting sessions, as in the figures) and pose_states.csv
    (state names, exclusion flag, feature means over the fitting sessions)."""
    m = pickle.load(open(f"{out_dir}/models_K_sweep.pkl", "rb"))
    hmm, mu, sd = m["models"][(final_K, final_kappa)], m["mu"], m["sd"]
    feats = list(m.get("features", features))
    k = int(fs / fs_fit)
    cond = {}
    for f in glob.glob(f"{pose_root}/*/*/*_pose_top_v1_8.h5"):
        dose, part = f.split("/")[-3:-1]
        if (dose, part) in cond_of:
            cond[os.path.basename(f).split("_pose_top")[0]] = cond_of[(dose, part)]
    have = {os.path.basename(f)[9:-7] for f in glob.glob(f"{out_dir}/features_*.csv.gz")}
    for s in load_all(only=set(cond) - have):
        s["feat"].to_csv(f"{out_dir}/features_{s['session']}.csv.gz", index=False)
        print(f"features of {s['session']} ({s['condition']})")
    sessions = []
    for f in sorted(glob.glob(f"{out_dir}/features_*.csv.gz")):
        feat = pd.read_csv(f)
        z = np.repeat(label(hmm, dict(feat=feat), mu, sd, feats), k)
        sessions.append((os.path.basename(f)[9:-7], feat.iloc[:len(z)], z))
    fit = [cond.get(n) in fit_conditions for n, _, _ in sessions]
    allf = pd.concat([f for (_, f, _), u in zip(sessions, fit) if u])
    allz = np.concatenate([z for (_, _, z), u in zip(sessions, fit) if u])
    prof = allf.groupby(allz).mean(numeric_only=True)
    order = prof.speed.sort_values().index.tolist()
    remap = np.argsort(order)                                  # fitted state -> speed rank
    for name, feat, z in sessions:
        zz = remap[z]
        st = np.flatnonzero(np.diff(np.r_[-1, zz]) != 0)
        en = np.r_[st[1:], len(zz)]
        clean = zz.copy()
        for a_, b_ in zip(st, en):
            if (b_ - a_) / fs < min_bout_s:
                clean[a_:b_] = -1
        pd.DataFrame(dict(frame=np.arange(len(zz)), state=zz, state_clean=clean)).to_csv(
            f"{out_dir}/states_{name}.csv.gz", index=False)
    prof = prof.loc[order].reset_index(drop=True)
    prof.insert(0, "occupancy", np.bincount(remap[allz], minlength=final_K) / len(allz))
    prof.insert(0, "excluded", [n in excluded_states for n in state_names])
    prof.insert(0, "label", [label_of[n] for n in state_names])
    prof.insert(0, "name", state_names)
    prof.index.name = "state"
    prof.to_csv(f"{out_dir}/pose_states.csv")
    print(prof[["name", "excluded", "occupancy", "speed", "turn", "head_speed",
                "abs_head_turn"]].round(3).to_string())
    print(f"saved {len(sessions)} state sequences and {out_dir}/pose_states.csv")


def session_summary():
    """Per-session behaviour over the imaging period from the stored features in out_dir:
    distance travelled (m), share of time immobile (torso speed < immobile_cm_s), mean
    speed; -> session_behavior.csv (e.g. to flag sedated sessions in later analyses)."""
    cond = {}
    for f in glob.glob(f"{pose_root}/*/*/*_pose_top_v1_8.h5"):
        dose, part = f.split("/")[-3:-1]
        if (dose, part) in cond_of:
            cond[os.path.basename(f).split("_pose_top")[0]] = cond_of[(dose, part)]
    rows = []
    for f in sorted(glob.glob(f"{out_dir}/features_*.csv.gz")):
        name = os.path.basename(f)[9:-7]
        v = pd.read_csv(f, usecols=["speed"]).speed.values
        rows.append(dict(session=name, mouse=re.search(r"_(m\d+|f\d+)_", name).group(1),
                         condition=cond.get(name), minutes=len(v) / fs / 60,
                         distance_m=v.sum() / fs / 100, mean_speed_cm_s=v.mean(),
                         immobile_share=np.mean(v < immobile_cm_s)))
    d = pd.DataFrame(rows)
    d.to_csv(f"{out_dir}/session_behavior.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.groupby("condition")[["distance_m", "mean_speed_cm_s", "immobile_share"]]
          .describe().round(2).T.to_string())
    print(f"saved {out_dir}/session_behavior.csv")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1:]
    export_states() if cmd == ["export"] else session_summary() if cmd == ["summary"] else main()
