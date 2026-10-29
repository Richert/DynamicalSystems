"""Keypoint-based behavioural states: features from the MARS keypoints + sticky AR-HMM.

Per session (imaging period only, aligned with align_pose_to_imaging.py), from the 7
keypoints in cm (session cm/px scale), smoothed (Gaussian, smooth_s) against keypoint
jitter, at 20 Hz:
  log_speed   log10(torso speed + speed_eps), torso = mean of neck, hips, tail base
  turn        |angular speed| of the body heading (tail base -> neck), deg/s
  log_turn    log10(turn + turn_eps) (used for the HMM; turn is heavy-tailed)
  head_angle  |angle| of the head (neck -> nose) relative to the body axis, deg
  body_len    nose - tail base length relative to the session median
  wall_dist   torso distance to the arena wall (cm; stored, not used for the HMM)
Absolute values for turn / head angle: left and right are not split into states.

The HMM features are z-scored globally, downsampled to fs_fit (block means), and a
sticky AR-HMM (ssm) is fitted for each K in K_grid on a training split of sessions
(stratified by condition, at most fit_cap_s per session); K is judged by held-out
log-likelihood per frame. Outputs (pose_arhmm/): per-session features (20 Hz), the
K sweep, the fitted models and a figure with the held-out curve and, per candidate K,
the state feature profiles, dwell times and occupancy per drug condition.

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
speed_eps = 0.5                    # cm/s, offset of the log speed (rest threshold scale)
turn_eps = 5.0                     # deg/s, offset of the log turning speed (heavy tail)
features = ["log_speed", "log_turn", "head_angle", "body_len"]
conditions = ["Control", drug, "Amphetamine", f"A + {drug}"]
cond_of = {("Vehicle", "Control"): "Control", ("Vehicle", "Amph"): "Amphetamine",
           ("HighDose", "Control"): drug, ("HighDose", "Amph"): f"A + {drug}"}

# AR-HMM
fs_fit = 10.0                      # fit at 10 Hz (block means of 2 frames)
fit_cap_s = 900                    # at most 15 min per session in the fit
K_grid = [3, 4, 5, 6, 7, 8, 10, 12]
ar_lags = 2                        # 200 ms of context at 10 Hz
kappa = 100.0
em_iters = 50
n_restarts = 3
heldout_frac = 0.2
n_jobs = int(os.environ.get("N_PROCESSES", 8))
seed = 1
show_K = [4, 5, 6, 8]              # candidate K values summarised in the figure

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
        nodes = [n.decode() for n in h["node_names"][...]]
    kp = np.transpose(tr, (2, 1, 0)) * scale                  # (T, 7, 2) in cm
    # arena centre from the keypoint extent over the whole video (as in the paper)
    lo = np.nanpercentile(kp.reshape(-1, 2), 0.01, axis=0)
    hi = np.nanpercentile(kp.reshape(-1, 2), 99.99, axis=0)
    centre = (lo + hi) / 2
    kp = kp[off:off + n_frames]
    kp = gaussian_filter1d(kp, sigma=smooth_s * fs, axis=0, mode="nearest")
    ix = {n: i for i, n in enumerate(nodes)}
    nose, neck, tail = kp[:, ix["nose"]], kp[:, ix["neck"]], kp[:, ix["tail_base"]]
    torso = kp[:, [ix["neck"], ix["hip1"], ix["hip2"], ix["tail_base"]]].mean(1)
    speed = np.r_[0.0, np.linalg.norm(np.diff(torso, axis=0), axis=1)] * fs
    body = neck - tail
    heading = np.unwrap(np.arctan2(body[:, 1], body[:, 0]))
    turn = np.abs(np.degrees(np.gradient(heading)) * fs)
    head = np.abs(angle_between(body, nose - neck))
    length = np.linalg.norm(nose - tail, axis=1)
    return pd.DataFrame(dict(
        log_speed=np.log10(speed + speed_eps), speed=speed,
        turn=gaussian_filter1d(turn, smooth_s * fs), head_angle=head,
        log_turn=np.log10(gaussian_filter1d(turn, smooth_s * fs) + turn_eps),
        body_len=length / np.median(length),
        wall_dist=arena_cm / 2 - np.linalg.norm(torso - centre, axis=1)))

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
    K, s = task
    npr.seed(s)
    hmm = ssm.HMM(K, len(features), observations="ar", transitions="sticky",
                  observation_kwargs=dict(lags=ar_lags), transition_kwargs=dict(kappa=kappa))
    hmm.fit(_DATA["train"], method="em", num_iters=em_iters, verbose=0)
    n_test = sum(len(x) for x in _DATA["test"])
    return K, float(hmm.log_likelihood(_DATA["train"])), \
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

    tasks = [(K, int(rng.integers(0, 2 ** 31))) for K in K_grid for _ in range(n_restarts)]
    with ProcessPoolExecutor(n_jobs, initializer=_init, initargs=(train, held)) as ex:
        res = list(ex.map(_fit, tasks))
    best = {}
    for K, ll_tr, ll_te, hmm in res:
        if K not in best or ll_tr > best[K][0]:
            best[K] = (ll_tr, ll_te, hmm)
    sweep = pd.DataFrame([dict(K=K, train_ll=v[0], heldout_ll_per_frame=v[1])
                          for K, v in sorted(best.items())])
    sweep.to_csv(f"{out_dir}/K_sweep.csv", index=False)
    print(sweep.round(4).to_string(index=False))
    pickle.dump(dict(models={K: v[2] for K, v in best.items()}, mu=mu, sd=sd, features=features,
                     fs_fit=fs_fit, test_sessions=[sessions[i]["session"] for i in sorted(test)]),
                open(f"{out_dir}/models_K_sweep.pkl", "wb"))
    summarise(sessions, {K: best[K][2] for K in show_K if K in best}, mu, sd, sweep)

# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def label(hmm, s, mu, sd):
    """Viterbi states at fs_fit for a whole session, renumbered later by speed."""
    X = np.ascontiguousarray(to_fit_rate((s["feat"][features].values - mu) / sd))
    return hmm.most_likely_states(X)

def summarise(sessions, models, mu, sd, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2 = "#1f1f1e", "#5f5e5a"
    colors = dict(zip(conditions, ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]))
    rows = []
    fig, axes = plt.subplots(1 + len(models), 3, figsize=(17, 3.4 * (1 + len(models))),
                             layout="constrained", squeeze=False)
    ax = axes[0, 0]
    ax.plot(sweep.K, sweep.heldout_ll_per_frame, "o-", color=ink)
    ax.set(xlabel="K", ylabel="held-out log-likelihood / frame", title="K sweep (held-out sessions)")
    for a in axes[0, 1:]:
        a.axis("off")
    raw_cols = ["speed", "turn", "head_angle", "body_len", "wall_dist"]
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
                rows.append(dict(K=K, state=k, session=s["session"], mouse=s["mouse"],
                                 condition=s["condition"], occupancy=np.mean(z == k)))
        # (a) state profiles (z-scored across states, raw means annotated)
        ax = axes[r, 0]
        zprof = (prof - prof.mean()) / (prof.std() + 1e-12)
        ax.imshow(zprof.T.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        for i in range(K):
            for j, c in enumerate(raw_cols):
                ax.text(i, j, f"{prof.iloc[i][c]:.2g}", ha="center", va="center", fontsize=7)
        ax.set_yticks(range(len(raw_cols)), ["speed cm/s", "turn deg/s", "head deg",
                                             "body len (rel)", "wall dist cm"], fontsize=8)
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
    fig.suptitle("Keypoint-feature AR-HMM: K sweep and candidate state sets (states ordered by speed)",
                 fontsize=11, color=ink)
    fig.savefig(f"{out_dir}/pose_arhmm_K_candidates.png", dpi=120, facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(f"{out_dir}/state_occupancy_candidates.csv", index=False)
    print(f"saved {out_dir}/pose_arhmm_K_candidates.png")


if __name__ == "__main__":
    main()
