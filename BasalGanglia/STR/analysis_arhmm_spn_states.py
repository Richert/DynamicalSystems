"""AR-HMM behavioral states x SPN firing-rate dynamics.

Pipeline (refined from the discussion):
  1. Build a 2-D behavioural observation [speed, acceleration] per frame from the
     animal velocity trace (from the .mat files), globally standardised.
  2. Fit ONE sticky AR-HMM to all sessions pooled (balanced by duration), so the
     discovered states are shared and therefore comparable across animals and
     drug conditions. AR order supplies the temporal ("delay-embedding") context.
  3. Viterbi-label every frame of every session; relabel states by ascending
     mean speed for interpretability.
  4. Estimate an animal-specific lag between SPN population rate and behaviour
     (cross-correlation), and apply it when reading neural activity per state.
  5. Per (session, state): compute the mean and SD of the SPN log-rates (= ERF
     CDF centre and 1/slope) and the mean pairwise correlation of the active neurons on
     subsamples of a common neuron count and frame count (averaged over draws),
     so the estimates are comparable.
  6. Per (session, transition type): the same readouts in the pooled pre- and
     post-transition windows and their change (post - pre); plus peri-transition
     SPN population-rate traces.
  7. Mixed-effects statistics (value ~ state*condition + (1|animal) and
     d_value ~ transition*condition + (1|animal)) + occupancy / transition rates,
     per drug and separately for D1 and D2 SPN animals. All model terms are
     written to arhmm_spn_model_terms.csv.

Run with the dedicated env (has ssm + statsmodels):
  /home/rgast/conda/envs/arhmm/bin/python analysis_arhmm_spn_states.py
"""

import os
import pickle
import warnings
import numpy as np
import h5py

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

neural_path = "/mnt/kennedy_labdata/Parkerlab/neural_data"   # server copy (10 drugs)
h5_path = "/home/rgast/data/parker_data/oasis_rates.h5"
# local cache of the speed traces (filled on first use from the .mat files)
speed_cache_path = "/home/rgast/data/parker_data/speed_traces_5hz.h5"
save_dir = "/home/rgast/data/parker_data"
out_dir = f"{save_dir}/arhmm_all10/K3"   # outputs of main() (model, CSVs, pickles, figures)
out_prefix = "arhmm"         # output file prefix; the embedding script sets "embed"
method_name = "AR-HMM"       # state-labelling method, used in figure titles
speed_field = "speed_traces_5hz"
fs = 5.0

# which drugs / conditions to include
# all drugs on the server (the 4-drug subset run -- haloperidol, SCH39166,
# SKF38393, MP10 -- is in arhmm_subset/K4)
drugs = ["haloperidol", "olanzapine", "clozapine", "SCH23390", "SCH39166", "SKF38393",
         "MP10", "xanomeline", "M4PAM", "SEP363856"]
# map (dose, raw condition) -> analysis label; None -> drop the session
condition_map = {
    ("Vehicle", "veh"): "Control",
    ("Vehicle", "amph"): "Amphetamine",
    ("HighDose", "amph"): "A + Drug",
    ("HighDose", "veh"): "Drug",
}                            # LowDose sessions are not used
conditions = ["Control", "Amphetamine", "A + Drug", "Drug"]
ref_condition = "Control"    # reference level of the condition factor in the models
spn_types = ["D1", "D2"]     # analysed separately (separate animal populations); HMM is shared

# AR-HMM
# load the existing arhmm_model.pkl instead of training, and only redo the
# post-training analysis (labelling/readouts/stats/plots) -- e.g. after changing
# min_state_posterior below. The saved model is not overwritten in this mode.
load_model = False
K_states = 3              # number of HMM states (set select_K=True to sweep)
ar_lags = 5              # AR order (temporal context; ~0.6 s at 5 Hz)
kappa = 100.0             # stickiness (self-transition prior); larger -> more persistent
em_iters = 50
n_restarts = 10           # EM restarts; keep the highest-likelihood fit
fit_cap_frames = 4500    # per-session cap for the *fit* (balances long amph sessions)
# K=5 fixed from the held-out sweep (all drugs, 2026-10-22): LL/frame keeps rising
# with K (3: 9.197, 4: 9.291, 5: 9.327, 6: 9.350, 7: 9.362, 8: 9.371) with an elbow
# at 4-5; K=8 gave near-duplicate, short-lived states and 23% unspecified frames
select_K = False        # if True, sweep K_grid to pick K
K_grid = [3, 4, 5, 6, 7, 8]   # extends past 6 so the selected K is not just the grid edge
# model selection: "heldout" (LL/frame on held-out whole sessions) | "bic" | "aic".
# BIC/AIC treat the strongly autocorrelated frames as independent samples and
# then tend to pick the largest K in K_grid.
k_criterion = "heldout"
heldout_frac = 0.2       # fraction of sessions held out for the "heldout" criterion
k_sweep_jobs = None      # parallel processes for the fits (None -> cpu_count; 1 -> serial)
# frames whose most-likely state has a posterior below this cutoff are left
# UNSPECIFIED (state id -1) instead of being forced into a state (0 -> disabled)
min_state_posterior = 0.6
seed = 1

# SPN readouts
rate_lims = (1e-6, 1e-3)     # winsorisation for the log-rate CDF (as in the figures)
# per-neuron amplitude normalisation of the rates: None (raw OASIS units) or
# "median_event" (divide by the neuron's median event amplitude -> typical events/s;
# removes brightness differences between neurons). Events are rates above
# event_rel_thresh x the neuron's max (below that is floating-point dust ~1e-12
# and smaller of the CNMF-E/OASIS output). rate_lims must then be in events/s.
rate_norm = None
event_rel_thresh = 1e-8
n_neurons_common = 50        # subsample neurons to this many per draw
# Readouts of the per-neuron mean rates of a draw (log10, winsorised to rate_lims):
#   mean_log, sd_log -- mean and SD over ALL sampled neurons (= the maximum-likelihood
#     ERF/Gaussian-CDF fit: centre and 1/slope). Most SPNs are silent over short
#     windows and sit at the rate_lims[0] floor, so both also reflect the fraction
#     of active neurons -- with this sparse spiking the two cannot be disentangled.
#   corr_active -- mean pairwise Pearson correlation (over the draw's frames) among
#     the ACTIVE neurons only (mean rate > rate_lims[0]); a simplified, size-
#     independent measure of population coordination. NaN for draws with
#     < min_active active neurons.
# (Dimensionality / participation ratio was dropped: with sparse firing it tracks
# the number of active neurons, and correlation-based versions show no effects.
# The skewness of the active neurons' log-rates showed no effects either.)
min_active = 10
# ALL readouts use exactly T_readout frames per draw: the fraction of silent
# neurons depends strongly on the number of frames averaged, so equal frame
# counts keep states/transitions/conditions comparable.
T_readout = 100              # frames per subsample (20 s); a state/transition needs >= this many
n_draws = 20                 # neuron x frame subsamples averaged per readout
readouts = ["mean_log", "sd_log", "corr_active"]

# animal-specific behaviour<->neural lag
max_lag_frames = 10          # search +/- this many frames (2 s at 5 Hz)

# transition-aligned analysis
trans_halfwin = 15           # +/- frames around a transition (3 s); also the pre/post readout window
min_dwell = 5                # min dwell (frames) on both sides to count a transition
max_gap = 5                  # bridge UNSPECIFIED (-1) gaps up to this many frames between two states

# statistics / output
min_sessions_per_cell = 3    # need this many animals with data to test a (state,condition) cell
max_sessions = None          # cap total sessions loaded (None = all); for quick tests
plot_results = True

# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

mice = {"D1": ["m085", "m040", "m298", "m404", "f487", "f694", "f857", "f859", "m794", "m797",
               "m795", "m973", "m974", "m659", "m975", "f976", "f977", "f979"],
        "D2": ["m971", "m972", "m106", "m120", "m377", "m380", "f414", "f480", "m483", "m485",
               "m241", "m242", "m523", "f605", "f808", "f811", "f840"]}
d1_set, d2_set = set(mice["D1"]), set(mice["D2"])

def spn_type(mouse):
    return "D1" if mouse in d1_set else ("D2" if mouse in d2_set else None)

def load_speed(drug, dose, session, cond_raw, cache):
    """Speed trace of one session from the local cache (h5 group drug/dose/session);
    on a miss it is read from the .mat file (slow on the network mount: the files
    hold all traces) and written to the cache. Returns None if unavailable."""
    key = f"{drug}/{dose}/{session}"
    if cache is not None and key in cache:
        return cache[key][...].astype(float)
    mat = f"{neural_path}/{drug}/{dose}/{session}/{cond_raw}_drug.mat"
    if not os.path.isfile(mat):
        return None
    try:
        from scipy.io import loadmat
        speed = np.asarray(loadmat(mat, simplify_cells=True)[f"{cond_raw}_drug"][speed_field],
                           dtype=float)
    except Exception:
        return None
    if cache is not None and cache.mode != "r":   # read-only if another run holds the file
        cache.create_dataset(key, data=speed)
        cache[key].attrs["source"] = mat
    return speed

def open_speed_cache():
    """Speed cache for writing; if another process holds it (parallel runs), open it
    read-only without HDF5 file locking (cache misses are then not stored). If even
    that fails (file mid-write), run without the cache (null context -> None)."""
    from contextlib import nullcontext
    try:
        return h5py.File(speed_cache_path, "a")
    except OSError:
        try:
            return h5py.File(speed_cache_path, "r", locking=False)
        except OSError:
            print("speed cache unavailable (in use); reading speed from the .mat files")
            return nullcontext()

def load_sessions():
    """Load OASIS rates (h5) + speed (local cache, else mat), aligned per session."""
    sessions = []
    with h5py.File(h5_path, "r") as h5, open_speed_cache() as cache:
        paths = []
        h5.visititems(lambda n, o: paths.append(n)
                      if (isinstance(o, h5py.Dataset) and n.endswith("rates")) else None)
        for name in sorted(paths):
            g = h5[name].parent
            a = g.attrs
            drug, dose, cond_raw, mouse = (str(a["drug"]), str(a["dose"]),
                                           str(a["condition"]), str(a["mouse"]))
            if drug not in drugs:
                continue
            cond = condition_map.get((dose, cond_raw))
            spn = spn_type(mouse)
            if cond is None or spn is None:
                continue
            session = str(a["session"])
            speed = load_speed(drug, dose, session, cond_raw, cache)
            if speed is None:
                continue
            rates = h5[name][...].astype(float)                  # (N, T)
            if rate_norm == "median_event":
                rates = normalize_by_event_amplitude(rates)
            T = min(rates.shape[1], len(speed))
            rates, speed = rates[:, :T], speed[:T]
            accel = np.gradient(speed)
            sessions.append(dict(drug=drug, dose=dose, condition=cond, mouse=mouse, spn=spn,
                                 session=session, rates=rates, speed=speed,
                                 obs=np.column_stack([speed, accel])))
            if max_sessions and len(sessions) >= max_sessions:
                break
    print(f"Loaded {len(sessions)} sessions "
          f"({len(set(s['mouse'] for s in sessions))} mice, drugs={drugs})")
    return sessions

def normalize_by_event_amplitude(rates):
    """Rates in units of the neuron's typical event per second: divide by the
    median event amplitude (rate units) and multiply by fs. All-zero rows stay zero."""
    out = np.zeros_like(rates)
    for i, r in enumerate(rates):
        if r.max() > 0:
            unit = np.median(r[r > event_rel_thresh * r.max()])
            out[i] = r / unit * fs
    return out

def standardize(sessions):
    """Global z-score of [speed, accel] (shared scale -> comparable states)."""
    allobs = np.concatenate([s["obs"] for s in sessions], axis=0)
    mu, sd = allobs.mean(0), allobs.std(0) + 1e-12
    for s in sessions:
        s["obs_z"] = (s["obs"] - mu) / sd
    return mu, sd

# ---------------------------------------------------------------------------
# AR-HMM
# ---------------------------------------------------------------------------

def fit_arhmm_once(datas, K, seed):
    """One EM run of a sticky AR-HMM from a seeded random initialisation."""
    import ssm
    import numpy.random as npr
    npr.seed(seed)
    hmm = ssm.HMM(K, 2, observations="ar", transitions="sticky",
                  observation_kwargs=dict(lags=ar_lags), transition_kwargs=dict(kappa=kappa))
    hmm.fit(datas, method="em", num_iters=em_iters, verbose=0)
    return hmm

def fit_arhmm(datas, K, rng):
    """Fit a sticky AR-HMM with n_restarts EM restarts (in parallel); keep the
    best-likelihood fit."""
    tasks = [(K, int(rng.integers(0, 2 ** 31))) for _ in range(n_restarts)]
    res = _parallel_fits(datas, datas, tasks)
    return max(res, key=lambda r: r[1])[4]

def fit_datasets(sessions, rng):
    """Duration-balanced fitting sequences: cap each session to fit_cap_frames
    (contiguous random window) so long sessions don't dominate the likelihood."""
    datas = []
    for s in sessions:
        z = s["obs_z"]
        if len(z) > fit_cap_frames:
            start = rng.integers(0, len(z) - fit_cap_frames)
            z = z[start:start + fit_cap_frames]
        datas.append(np.ascontiguousarray(z))
    return datas

# module-level worker state/functions so the fits can run in a process pool
_SWEEP = {}

def _sweep_init(fit_data, eval_data):
    # one BLAS thread per worker: parallelism comes from the pool, not from BLAS
    for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    _SWEEP["fit"], _SWEEP["eval"] = fit_data, eval_data

def _fit_one(k_seed):
    """One EM restart for one K on the shared fit data; return (K, logL on fit
    data, logL on eval data, n_eval, hmm). The fit logL picks the best restart."""
    K, s = k_seed
    hmm = fit_arhmm_once(_SWEEP["fit"], K, s)
    ll_fit = float(hmm.log_likelihood(_SWEEP["fit"]))
    ll_eval = ll_fit if _SWEEP["eval"] is _SWEEP["fit"] else float(hmm.log_likelihood(_SWEEP["eval"]))
    return K, ll_fit, ll_eval, int(sum(len(d) for d in _SWEEP["eval"])), hmm

def _parallel_fits(fit_data, eval_data, tasks):
    """Run _fit_one over (K, seed) tasks in a process pool (k_sweep_jobs workers;
    None -> all CPUs, 1 -> serial)."""
    n_jobs = min(len(tasks), k_sweep_jobs or (os.cpu_count() or 1))
    if n_jobs <= 1:
        _sweep_init(fit_data, eval_data)
        return [_fit_one(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    print(f"  {len(tasks)} AR-HMM fits over {n_jobs} parallel processes ...")
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_sweep_init,
                             initargs=(fit_data, eval_data)) as ex:
        return list(ex.map(_fit_one, tasks))

def _effective_K(hmm, datas):
    """Number of states actually used (non-zero Viterbi occupancy) -- a fit that
    collapses to fewer states is scored at its effective K, so dead states cost
    nothing and don't inflate the selected K."""
    used = set()
    for d in datas:
        used.update(np.unique(hmm.most_likely_states(np.ascontiguousarray(d))).tolist())
    return len(used)

def _n_params(K):
    """Free parameters of the sticky AR-HMM: transitions + initial + per-state
    AR-Gaussian observations (D=2 obs dim, ar_lags AR order)."""
    D = 2
    per_state = D * D * ar_lags + D + D * (D + 1) // 2   # AR matrix + bias + covariance
    return K * (K - 1) + (K - 1) + K * per_state          # transitions + initial + obs

def choose_K(sessions, rng):
    """Sweep K_grid and pick K by BIC/AIC (penalise parameters, so dead states
    don't win) or held-out LL/frame. All (K, restart) fits run in parallel; the
    best restart per K (fit logL) is scored."""
    if k_criterion == "heldout":
        # hold out heldout_frac of the sessions of every (drug, condition) group, so
        # all groups are represented; test sessions are cropped like the fit data
        groups = {}
        for i, s in enumerate(sessions):
            groups.setdefault((s["drug"], s["condition"]), []).append(i)
        test = set()
        for idx in groups.values():
            n_test = max(1, int(round(heldout_frac * len(idx))))
            test.update(rng.choice(idx, n_test, replace=False).tolist())
        train = [i for i in range(len(sessions)) if i not in test]
        print(f"  held-out split: {len(train)} train / {len(test)} test sessions")
        fit_data = fit_datasets([sessions[i] for i in train], rng)
        eval_data = fit_datasets([sessions[i] for i in sorted(test)], rng)
    else:                                                # AIC/BIC: fit and score on all data
        fit_data = eval_data = fit_datasets(sessions, rng)
    # a seed per (K, restart)
    tasks = [(K, int(rng.integers(0, 2 ** 31))) for K in K_grid for _ in range(n_restarts)]
    res = _parallel_fits(fit_data, eval_data, tasks)

    best_fit = {}                                # K -> best restart (highest fit logL)
    for r in res:
        if r[0] not in best_fit or r[1] > best_fit[r[0]][1]:
            best_fit[r[0]] = r
    ll = {K: best_fit[K][2] for K in K_grid}     # eval logL of the best restart
    n = {K: best_fit[K][3] for K in K_grid}
    models = {K: best_fit[K][4] for K in K_grid}
    keff = {K: _effective_K(models[K], eval_data) for K in K_grid}   # dead states dropped
    if k_criterion == "heldout":
        score = {K: ll[K] / n[K] for K in K_grid}                  # higher is better
        best = max(score, key=score.get)
        for K in K_grid:
            print(f"  K={K} (eff {keff[K]}): held-out LL/frame = {score[K]:.4f}")
    else:
        # score on the EFFECTIVE parameter count, so a collapsed K is treated as its
        # effective K (dead states add no likelihood and cost no parameters here)
        pen = (lambda K: 2 * _n_params(keff[K])) if k_criterion == "aic" \
            else (lambda K: _n_params(keff[K]) * np.log(n[K]))
        score = {K: pen(K) - 2 * ll[K] for K in K_grid}            # lower is better
        best = min(score, key=score.get)
        for K in K_grid:
            print(f"  K={K} (eff {keff[K]}): logL={ll[K]:.0f}  params={_n_params(keff[K])}  "
                  f"{k_criterion.upper()}={score[K]:.0f}")
    print(f"  -> selected K={best} (effective K={keff[best]})")
    # for AIC/BIC the winning model was fit on the FULL data during the sweep, so
    # reuse it; for held-out it was fit on a train split only -> caller refits.
    best_hmm = models[best] if k_criterion != "heldout" else None
    return best, score, best_hmm

def relabel_by_speed(sessions, hmm, K):
    """Viterbi-label all frames; permute state ids to ascending mean speed.
    Frames whose most-likely-state posterior is below `min_state_posterior` are
    labelled -1 (UNSPECIFIED) and excluded from every state."""
    n_unspec = n_tot = 0
    for s in sessions:
        obs = np.ascontiguousarray(s["obs_z"])
        z = hmm.most_likely_states(obs)
        if min_state_posterior > 0:
            Ez = hmm.expected_states(obs)[0]                 # (T, K) posterior marginals
            conf = Ez[np.arange(len(z)), z]                  # posterior of the assigned state
            z = np.where(conf >= min_state_posterior, z, -1)
            n_unspec += int(np.sum(z < 0)); n_tot += len(z)
        s["z"] = z
    if min_state_posterior > 0:
        print(f"Unspecified (posterior < {min_state_posterior}): "
              f"{n_unspec}/{n_tot} frames ({100 * n_unspec / max(n_tot, 1):.1f}%)")
    # mean speed per state (original units); -1 frames excluded. States with no
    # assigned frames are DEAD and dropped: the survivors define the effective K
    # and are renumbered 0..K_eff-1 by ascending mean speed.
    sums = np.zeros(K); cnts = np.zeros(K)
    for s in sessions:
        for k in range(K):
            m = s["z"] == k
            sums[k] += s["speed"][m].sum(); cnts[k] += m.sum()
    used = np.where(cnts > 0)[0]
    if len(used) < K:
        print(f"Dropping {K - len(used)} dead state(s) (zero occupancy); effective K = {len(used)}")
    mean_speed = sums[used] / cnts[used]
    order = np.argsort(mean_speed)                     # order within surviving states
    remap = {int(used[o]): i for i, o in enumerate(order)}    # old id -> new 0..K_eff-1
    for s in sessions:
        z = s["z"]
        znew = np.full_like(z, -1)                     # dead / low-posterior frames stay -1
        for old, new in remap.items():
            znew[z == old] = new
        s["z"] = znew
    return mean_speed[order]                            # length = effective K

# ---------------------------------------------------------------------------
# animal-specific neural<->behaviour lag
# ---------------------------------------------------------------------------

def animal_lags(sessions):
    """Per animal: lag (frames) maximising xcorr between SPN population rate and
    speed. Convention: neural activity for behaviour frame f is rates[:, f - lag]."""
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    by_mouse = {}
    for s in sessions:
        pr = s["rates"].mean(0)
        v = s["speed"]
        pr = (pr - pr.mean()) / (pr.std() + 1e-12)
        v = (v - v.mean()) / (v.std() + 1e-12)
        cc = np.array([np.mean(pr[max(0, -d):len(pr) - max(0, d)] *
                               v[max(0, d):len(v) - max(0, -d)]) for d in lags])
        by_mouse.setdefault(s["mouse"], []).append(cc)
    lag_of = {}
    for mouse, ccs in by_mouse.items():
        mean_cc = np.mean(ccs, axis=0)
        lag_of[mouse] = int(lags[np.argmax(mean_cc)])
    return lag_of

# ---------------------------------------------------------------------------
# SPN readouts
# ---------------------------------------------------------------------------

def to_lograte(x):
    return np.log10(np.clip(x, *rate_lims))

def draw_stats(R):
    """(mean_log, sd_log, corr_active) of a neurons x frames draw, see `readouts`."""
    m = R.mean(axis=1)
    x = to_lograte(m)
    Ra = R[(m > rate_lims[0]) & (R.std(axis=1) > 0)]
    if len(Ra) >= min_active:
        C = np.corrcoef(Ra)
        rc = float(C[np.triu_indices(len(Ra), k=1)].mean())
    else:
        rc = np.nan
    return float(np.mean(x)), float(np.std(x)), rc

def neuron_sets(s, rng):
    """n_draws neuron subsets per session, drawn once from the non-degenerate
    (not all-zero) neurons and shared by all states/transitions of the session,
    so within-session comparisons are not confounded by different neurons."""
    if "neu_sets" not in s:
        alive = np.flatnonzero(s["rates"].any(axis=1))
        s["neu_sets"] = ([rng.choice(alive, n_neurons_common, replace=False) for _ in range(n_draws)]
                         if len(alive) >= n_neurons_common else None)
    return s["neu_sets"]

def neural_frames(s, frames, lag):
    """Lag-corrected neural frame indices for behaviour frames (out-of-range dropped)."""
    f = np.asarray(frames) - lag
    return f[(f >= 0) & (f < s["rates"].shape[1])]

def subsampled_readouts(s, frames, rng):
    """`readouts` averaged over n_draws subsamples of n_neurons_common neurons x
    T_readout frames; NaNs if too few neurons or frames."""
    neu_sets = neuron_sets(s, rng)
    if neu_sets is None or len(frames) < T_readout:
        return (np.nan,) * len(readouts)
    vals = []
    for neu in neu_sets:
        R = s["rates"][np.ix_(neu, rng.choice(frames, T_readout, replace=False))]
        vals.append(draw_stats(R))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # all-NaN columns -> NaN
        return tuple(float(v) for v in np.nanmean(vals, axis=0))

def state_readouts(sessions, lag_of, rng):
    """One row of `readouts` + occupancy per (session, state), lag-corrected."""
    rows = []
    for s in sessions:
        lag = lag_of.get(s["mouse"], 0)
        for k in np.unique(s["z"]):
            if k < 0:                      # skip UNSPECIFIED frames (low-posterior)
                continue
            frames = np.where(s["z"] == k)[0]
            nf = len(frames)
            row = dict(mouse=s["mouse"], condition=s["condition"], spn=s["spn"],
                       drug=s["drug"], session=s["session"], state=int(k),
                       occupancy=nf / len(s["z"]), n_frames=nf)
            row.update(zip(readouts, subsampled_readouts(s, neural_frames(s, frames, lag), rng)))
            rows.append(row)
    return rows

def find_transitions(z):
    """State switches frm->to between SPECIFIED states as (last frm frame, first
    to frame, frm, to). An UNSPECIFIED (-1) gap of <= max_gap frames between the
    two states is bridged; both states need >= min_dwell frames of dwell."""
    starts = np.r_[0, np.flatnonzero(np.diff(z)) + 1]
    ends = np.r_[starts[1:], len(z)]
    runs = list(zip(starts, ends, z[starts]))
    out = []
    for a in range(len(runs) - 1):
        s0, e0, frm = runs[a]
        if frm < 0 or e0 - s0 < min_dwell:
            continue
        b = a + 1
        if runs[b][2] < 0:                               # unspecified gap
            if runs[b][1] - runs[b][0] > max_gap or b + 1 >= len(runs):
                continue
            b += 1
        s1, e1, to = runs[b]
        if to < 0 or to == frm or e1 - s1 < min_dwell:
            continue
        out.append((int(e0 - 1), int(s1), int(frm), int(to)))
    return out

def transition_readouts(sessions, lag_of, rng):
    """One row per (session, transition type frm->to): the readouts of the
    pooled pre-transition window (the trans_halfwin frames ending at the last frm
    frame) and post-transition window (trans_halfwin frames from the first `to`
    frame) over all occurrences, plus the change d_q = post - pre. Pre and post use
    the same neuron subsets and frame count, so d_q isolates the transition."""
    W = trans_halfwin
    rows = []
    for s in sessions:
        lag = lag_of.get(s["mouse"], 0)
        by_type = {}
        for t_last, t_first, frm, to in find_transitions(s["z"]):
            by_type.setdefault((frm, to), []).append((t_last, t_first))
        minutes = len(s["z"]) / fs / 60
        for (frm, to), occ in sorted(by_type.items()):
            pre = np.unique(np.concatenate([np.arange(t - W + 1, t + 1) for t, _ in occ]))
            post = np.unique(np.concatenate([np.arange(t, t + W) for _, t in occ]))
            pre = neural_frames(s, pre[pre >= 0], lag)
            post = neural_frames(s, post[post < len(s["z"])], lag)
            row = dict(mouse=s["mouse"], condition=s["condition"], spn=s["spn"],
                       drug=s["drug"], session=s["session"], transition=f"{frm}->{to}",
                       frm=frm, to=to, n_trans=len(occ), trans_per_min=len(occ) / minutes,
                       n_frames_pre=len(pre), n_frames_post=len(post))
            pre_q = subsampled_readouts(s, pre, rng)
            post_q = subsampled_readouts(s, post, rng)
            for q, a, b in zip(readouts, pre_q, post_q):
                row[f"{q}_pre"], row[f"{q}_post"], row[f"d_{q}"] = a, b, b - a
            rows.append(row)
    return rows

# ---------------------------------------------------------------------------
# transition-aligned analysis
# ---------------------------------------------------------------------------

def transition_traces(sessions, lag_of):
    """Peri-transition SPN population rate, grouped by (from,to,condition).
    Returns per-animal averaged traces to avoid pseudoreplication."""
    W = trans_halfwin
    acc = {}   # (frm,to,condition) -> {mouse: [traces]}
    for s in sessions:
        z = s["z"]
        pr = s["rates"].mean(0)
        pr = (pr - pr.mean()) / (pr.std() + 1e-12)
        lag = lag_of.get(s["mouse"], 0)
        for _, t, frm, to in find_transitions(z):       # aligned to the first `to` frame
            a, b = t - W - lag, t + W + 1 - lag
            if a < 0 or b > len(pr):
                continue
            key = (frm, to, s["condition"])
            acc.setdefault(key, {}).setdefault(s["mouse"], []).append(pr[a:b])
    # average within mouse, then across mice
    out = {}
    for key, per_mouse in acc.items():
        mouse_means = [np.mean(v, axis=0) for v in per_mouse.values() if len(v)]
        if len(mouse_means) >= 2:
            M = np.array(mouse_means)
            out[key] = dict(mean=M.mean(0), sem=M.std(0) / np.sqrt(len(M)),
                            n_mice=len(M), n_trans=sum(len(v) for v in per_mouse.values()))
    return out

# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def run_stats(df, factor="state", quantities=tuple(readouts)):
    """Mixed-effects test per quantity. Keeps only (factor,condition) cells with
    enough animals, then tries interaction -> additive -> OLS, falling back on
    singular designs (empty interaction cells are common when a state is rare in
    a condition). Returns the p-values of whichever model fitted, FDR-corrected.
    `factor` is "state" for state readouts or "transition" for transition readouts."""
    import statsmodels.api as sm
    from patsy import dmatrices
    from statsmodels.stats.multitest import multipletests
    from pandas import Categorical
    f = factor

    def fit_one(q):
        d = df.dropna(subset=[q]).copy()
        cell = d.groupby([f, "condition"])["mouse"].nunique()
        good = cell[cell >= min_sessions_per_cell].index
        d = d[d.set_index([f, "condition"]).index.isin(good)].copy()
        # keep only factor levels/conditions still present in >=2 of the other factor
        for _ in range(3):
            sc = d.groupby(f)["condition"].nunique()
            d = d[d[f].isin(sc[sc >= 2].index)]
            cs = d.groupby("condition")[f].nunique()
            d = d[d["condition"].isin(cs[cs >= 2].index)]
        # treatment contrasts are "vs Control": levels without Control data cannot be
        # compared and are dropped
        if ref_condition in set(d["condition"]):
            d = d[d[f].isin(d.loc[d["condition"] == ref_condition, f].unique())]
        if d[f].nunique() < 2 or d["condition"].nunique() < 2 or len(d) < 8:
            return None
        # reference level of the factor = the one present in the most conditions,
        # so condition main effects (= condition effect in the reference level)
        # are identifiable; ties -> lowest state / most frequent transition
        cover = d.groupby(f)["condition"].nunique()
        order = sorted(cover.index, key=lambda l: (-cover[l], -int((d[f] == l).sum()), l))
        d[f] = Categorical(d[f], categories=[order[0]] + sorted(order[1:]))
        present = [c for c in conditions if c in set(d["condition"])]
        if ref_condition in present:
            present = [ref_condition] + [c for c in present if c != ref_condition]
        d["condition"] = Categorical(d["condition"], categories=present)
        # per design: mixed model first, then OLS with mouse-clustered robust SEs
        # (guards pseudoreplication) if the mixed fit is degenerate; drop the
        # interaction only if neither fits
        last_err = "no valid fit"
        for design, rhs in [("interaction", f"C({f})*C(condition)"), ("additive", f"C({f})+C(condition)")]:
            y, X = dmatrices(f"{q} ~ {rhs}", d, return_type="dataframe")
            X = X.loc[:, (X != 0).any(axis=0)]        # interaction columns of empty cells
            if np.linalg.matrix_rank(X.values) < X.shape[1]:
                last_err = f"{design}: rank-deficient design (missing cells)"
                continue
            for kind in ["mixedlm", "ols-clustered"]:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if kind == "mixedlm":
                            mdf = _fit_mixedlm(y, X, d["mouse"].values)
                        else:
                            mdf = sm.OLS(y, X).fit(cov_type="cluster",
                                                   cov_kwds={"groups": d["mouse"].astype("category").cat.codes.values})
                    ok, why = _valid_fit(mdf, kind, d["mouse"].nunique())
                    if ok:
                        return d, mdf, f"{kind}-{design}"
                    last_err = f"{kind}-{design}: {why}"
                except Exception as e:
                    last_err = f"{kind}-{design}: {e}"
        return ("error", last_err)

    results = {}
    for q in quantities:
        r = fit_one(q)
        if r is None:
            results[q] = None
        elif len(r) == 2:            # ("error", message)
            results[q] = dict(error=r[1])
        else:                        # (dataframe, model, kind)
            _, mdf, kind = r
            pvals = mdf.pvalues.drop(labels=[i for i in mdf.pvalues.index
                                             if i in ("Intercept", "Group Var")], errors="ignore")
            rej, q_adj, *_ = multipletests(pvals.values, method="fdr_bh")
            # params + covariance are kept so arbitrary contrasts (e.g. A + Drug vs
            # Amphetamine) can be computed later without refitting
            results[q] = dict(model=mdf, kind=kind, params=mdf.params, cov=mdf.cov_params(),
                              n_obs=int(mdf.nobs), n_mice=int(r[0]["mouse"].nunique()),
                              reference=(r[0][f].cat.categories[0], r[0]["condition"].cat.categories[0]),
                              fdr={k: (float(p), float(qa), bool(rj))
                                   for k, p, qa, rj in zip(pvals.index, pvals.values, q_adj, rej)})
    return results

def _fit_mixedlm(y, X, groups):
    """Random-intercept ML fit; lbfgs sometimes hits a singular matrix where other
    optimisers converge to the same optimum, so try several in turn."""
    import statsmodels.api as sm
    mdf, err = None, None
    for method in ["lbfgs", "bfgs", "powell", "nm"]:
        try:
            mdf = sm.MixedLM(y, X, groups=groups).fit(reml=False, method=method)
            if mdf.converged:
                return mdf
        except Exception as e:
            err = e
    if mdf is None:
        raise err
    return mdf                                   # not converged -> rejected by _valid_fit

def _valid_fit(mdf, kind, n_mice):
    """Reject degenerate fits: non-converged mixed models, non-finite p-values or
    non-positive SEs of the fixed effects (singular Hessian), and clustered OLS
    with at least as many parameters as clusters (unreliable robust covariance)."""
    fe = [i for i in mdf.pvalues.index if i != "Group Var"]
    p, se = mdf.pvalues[fe].values, mdf.bse[fe].values
    if kind == "mixedlm" and not mdf.converged:
        return False, "not converged"
    if kind == "ols-clustered" and len(fe) >= n_mice:
        return False, f"{len(fe)} parameters >= {n_mice} mouse clusters"
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(se))):
        return False, "non-finite p-values/SEs"
    if np.any(se <= 0):
        return False, "degenerate SEs (SE <= 0)"
    return True, ""

def transition_change_tests(df_t, quantities=tuple(f"d_{q}" for q in readouts)):
    """Does a transition change the readout? Per (transition, condition): one value
    per animal (mean over its sessions), one-sample t-test of post - pre vs 0;
    BH-FDR across all cells of a quantity. Cells need min_sessions_per_cell animals."""
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests
    from pandas import DataFrame
    rows = []
    for q in quantities:
        pa = df_t.dropna(subset=[q]).groupby(["transition", "condition", "mouse"])[q].mean()
        for (tr, c), v in pa.groupby(level=["transition", "condition"]):
            if len(v) >= min_sessions_per_cell:
                rows.append(dict(quantity=q, transition=tr, condition=c, n_mice=len(v),
                                 mean=v.mean(), sem=v.std(ddof=1) / np.sqrt(len(v)),
                                 p=ttest_1samp(v.values, 0.0).pvalue))
    out = DataFrame(rows)
    if len(out):
        out["q_fdr"] = np.nan
        for q, g in out.groupby("quantity"):
            ok = g["p"].notna()
            out.loc[g.index[ok], "q_fdr"] = multipletests(g["p"][ok], method="fdr_bh")[1]
    return out

def print_stats(stats):
    for q, r in stats.items():
        print(f"\n--- {q} ---")
        if r is None:
            print("  insufficient data")
        elif "error" in r:
            print("  model failed:", r["error"])
        else:
            print(f"  [{r['kind']}; reference = {r['reference'][0]} / {r['reference'][1]}; "
                  f"{r['n_mice']} mice]")
            for term, (p, qa, sig) in r["fdr"].items():
                print(f"  {term:40s} p={p:.4g}  q(FDR)={qa:.4g} {'*' if sig else ''}")

def stats_table(stats, drug, spn, analysis):
    """Flat table of all fixed-effect terms (coef, SE, p, FDR q) of run_stats output."""
    from pandas import DataFrame
    rows = []
    for q, r in stats.items():
        if r is None or "error" in r:
            continue
        for term, (p, qa, sig) in r["fdr"].items():
            rows.append(dict(drug=drug, spn=spn, analysis=analysis, quantity=q, model=r["kind"],
                             n_mice=r["n_mice"], n_obs=r["n_obs"], reference_level=r["reference"][0],
                             reference_condition=r["reference"][1], term=term,
                             coef=float(r["params"][term]), se=float(np.sqrt(r["cov"].loc[term, term])),
                             p=p, q_fdr=qa, significant=sig))
    return DataFrame(rows)

def picklable(stats):
    """Drop fitted model objects, keep the summaries."""
    return {q: (r if (r is None or "error" in r) else {k: v for k, v in r.items() if k != "model"})
            for q, r in stats.items()}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    from pandas import DataFrame
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    sessions = load_sessions()

    if load_model:
        # reuse the existing fit; redo only the post-training analysis
        m = pickle.load(open(f"{out_dir}/arhmm_model.pkl", "rb"))
        hmm, mu, sd, K, kscores = m["hmm"], m["mu"], m["sd"], m["K"], None
        for s in sessions:
            s["obs_z"] = (s["obs"] - mu) / sd
        print(f"Loaded existing AR-HMM from disk (K={K}, lags={m.get('ar_lags')}, "
              f"kappa={m.get('kappa')}); redoing post-training analysis only.")
    else:
        mu, sd = standardize(sessions)
        if select_K:
            print(f"Selecting K ({k_criterion}) ...")
            K, kscores, hmm = choose_K(sessions, rng)   # AIC/BIC keep the full-data fit
        else:
            K, kscores, hmm = K_states, None, None
        # --- ONE shared AR-HMM across all drugs (states comparable across drugs) ---
        if hmm is None:                                 # heldout sweep or no sweep -> fit now
            print(f"Fitting SHARED sticky AR-HMM (K={K}, lags={ar_lags}) on {len(sessions)} sessions ...")
            hmm = fit_arhmm(fit_datasets(sessions, rng), K, rng)
        else:
            print(f"Reusing the K={K} model fitted during the {k_criterion} sweep (no extra training run).")
        # persist the fitted model so downstream scripts reuse this EXACT fit
        pickle.dump(dict(hmm=hmm, mu=mu, sd=sd, K=K, ar_lags=ar_lags, kappa=kappa, drugs=drugs),
                    open(f"{out_dir}/arhmm_model.pkl", "wb"))
        print(f"Saved fitted AR-HMM to {out_dir}/arhmm_model.pkl")

    state_speed = relabel_by_speed(sessions, hmm, K)     # applies min_state_posterior cutoff
    analyze_states(sessions, state_speed, K, kscores, rng)

def analyze_states(sessions, state_speed, K, kscores, rng):
    """Everything downstream of the per-frame state labels s["z"] (speed-ordered,
    -1 = unspecified): lags, state + transition readouts, per-(drug, SPN type)
    statistics, tables and figures. Shared by the AR-HMM and embedding pipelines."""
    from pandas import DataFrame
    os.makedirs(out_dir, exist_ok=True)
    print("State mean speed (cm/s), ascending:",
          np.array2string(state_speed, precision=2))

    lag_of = animal_lags(sessions)
    print("Animal lags (frames):", {k: lag_of[k] for k in list(lag_of)[:8]}, "...")

    df = DataFrame(state_readouts(sessions, lag_of, rng))
    df.to_csv(f"{out_dir}/{out_prefix}_spn_state_readouts.csv", index=False)  # drug-labelled
    df_t = DataFrame(transition_readouts(sessions, lag_of, rng))
    df_t.to_csv(f"{out_dir}/{out_prefix}_spn_transition_readouts.csv", index=False)

    # --- per-drug analysis, separately for D1 and D2 SPNs (separate animal
    # populations), using the SHARED state labels ---
    from pandas import concat
    per_drug, change_tests, stat_tables = {}, [], []
    for drug in [d for d in drugs if d in set(df["drug"])]:
        for spn in spn_types:
            tag = f"{drug}_{spn}"
            df_d = df[(df["drug"] == drug) & (df["spn"] == spn)]
            dt_d = df_t[(df_t["drug"] == drug) & (df_t["spn"] == spn)]
            if df_d.empty:
                continue
            print(f"\n####################  drug = {drug}, {spn} SPNs "
                  f"({df_d['mouse'].nunique()} mice)  ####################")
            sess_d = [s for s in sessions if s["drug"] == drug and s["spn"] == spn]
            lag_d = {m: lag_of[m] for m in df_d["mouse"].unique() if m in lag_of}

            occ = df_d.groupby(["state", "condition"])["occupancy"].mean().unstack("condition")
            print("Mean occupancy by state x condition:\n", occ.round(3))
            trate = dt_d.groupby(["transition", "condition"])["trans_per_min"].mean().unstack("condition")
            print("\nMean transitions/min by type x condition:\n", trate.round(3))

            print("\n=== STATES: mixed-effects (value ~ state*condition + (1|animal)) ===")
            stats = run_stats(df_d)
            print_stats(stats)

            print("\n=== TRANSITIONS: mixed-effects on post-pre change "
                  "(d_value ~ transition*condition + (1|animal)) ===")
            tstats = run_stats(dt_d, factor="transition", quantities=tuple(f"d_{q}" for q in readouts))
            print_stats(tstats)
            stat_tables += [stats_table(stats, drug, spn, "state"),
                            stats_table(tstats, drug, spn, "transition")]

            tchg = transition_change_tests(dt_d)
            if len(tchg):
                tchg.insert(0, "spn", spn)
                tchg.insert(0, "drug", drug)
                change_tests.append(tchg)
                print("\n=== TRANSITIONS: post-pre change vs 0 (per-animal one-sample t, FDR) ===")
                print(tchg[tchg["q_fdr"] < 0.05].to_string(index=False, float_format="%.4g")
                      if (tchg["q_fdr"] < 0.05).any() else "  no significant changes (q<0.05)")

            trans = transition_traces(sess_d, lag_of)
            # store only picklable summaries (drop fitted model objects)
            per_drug.setdefault(drug, {})[spn] = dict(
                occupancy=occ, transition_rate=trate, transitions=trans, lag_of=lag_d,
                stats=picklable(stats), transition_stats=picklable(tstats), transition_change=tchg)
            if plot_results:
                make_plots(tag, df_d, state_speed, trans, occ, lag_d)
                plot_transition_readouts(tag, dt_d)

    if change_tests:
        concat(change_tests).to_csv(f"{out_dir}/{out_prefix}_spn_transition_change_tests.csv", index=False)
    # every model term of every (drug, spn, analysis, quantity): pick effects for figures from here
    concat(stat_tables).to_csv(f"{out_dir}/{out_prefix}_spn_model_terms.csv", index=False)
    # persist shared model summary + per-(drug, spn) results (input for the cross-drug script)
    pickle.dump(dict(state_speed=state_speed, K=K, kscores=kscores, lag_of=lag_of,
                     ar_lags=ar_lags, drugs=list(per_drug), per_drug=per_drug),
                open(f"{out_dir}/{out_prefix}_spn_results.pkl", "wb"))
    print(f"\nSaved readout CSVs and per-(drug, spn) results pickle to {out_dir}/")

# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def make_plots(drug, df, state_speed, trans, occ, lag_of):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sb
    sb.set_palette("colorblind")
    K = len(state_speed)
    cond_colors = condition_colors()

    # (1) state signatures (mean speed) + occupancy; (2) readouts; (3) lags
    fig, axes = plt.subplots(2, 4, figsize=(21, 9), layout="tight")
    axes[0, 3].axis("off")
    fig.suptitle(f"{method_name} behavioural states x SPN dynamics - {drug}", fontsize=15)

    axes[0, 0].bar(range(K), state_speed, color="0.5")
    axes[0, 0].set(xlabel="state (speed-ordered)", ylabel="mean speed (cm/s)",
                   title="state signatures (shared HMM)")

    occ_p = occ.reindex(columns=[c for c in conditions if c in occ.columns])
    occ_p.plot(kind="bar", ax=axes[0, 1], color=[cond_colors[c] for c in occ_p.columns])
    axes[0, 1].set(xlabel="state", ylabel="occupancy", title="occupancy by condition")

    axes[0, 2].hist(list(lag_of.values()), bins=range(-max_lag_frames, max_lag_frames + 2),
                    color="0.5")
    axes[0, 2].set(xlabel="lag (frames)", ylabel="# animals",
                   title="animal neural<->behaviour lag")

    for ax, q in zip(axes[1], readouts):
        g = df.dropna(subset=[q]).groupby(["state", "condition"])[q].agg(["mean", "sem"])
        for c in [c for c in conditions if c in df["condition"].unique()]:
            sub = g.xs(c, level="condition") if c in g.index.get_level_values("condition") else None
            if sub is None:
                continue
            ax.errorbar(sub.index, sub["mean"], yerr=sub["sem"], marker="o",
                        color=cond_colors[c], label=c, capsize=3)
        ax.set(xlabel="state", ylabel=q, title=f"SPN {q} by state x condition")
    for ax in axes[1, len(readouts):]:
        ax.axis("off")
    axes[1, 0].legend(fontsize=9)
    fig.savefig(f"{out_dir}/{out_prefix}_spn_states_summary_{drug}.png", dpi=120)
    plt.close(fig)

    # (3) transition-aligned population rate for the most-sampled transitions
    keys = sorted(trans, key=lambda k: -trans[k]["n_trans"])[:6]
    if keys:
        t = (np.arange(-trans_halfwin, trans_halfwin + 1)) / fs
        fig2, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.4),
                                 sharey=True, layout="tight")
        axs = np.atleast_1d(axs)
        for ax, key in zip(axs, keys):
            frm, to, cond = key
            d = trans[key]
            ax.plot(t, d["mean"], color=cond_colors.get(cond, "k"))
            ax.fill_between(t, d["mean"] - d["sem"], d["mean"] + d["sem"], alpha=0.3,
                            color=cond_colors.get(cond, "k"))
            ax.axvline(0, ls="--", c="0.5", lw=1)
            ax.set(title=f"S{frm}->S{to}\n{cond} (n={d['n_mice']}m)",
                   xlabel="time from transition (s)")
        axs[0].set_ylabel("pop. rate (z)")
        fig2.suptitle(f"transition-aligned SPN population rate - {drug}", fontsize=13)
        fig2.savefig(f"{out_dir}/{out_prefix}_spn_transitions_{drug}.png", dpi=120)
        plt.close(fig2)
    print(f"Saved figures to {out_dir}/{out_prefix}_spn_*_{drug}.png")

def condition_colors():
    import seaborn as sb
    return dict(zip(conditions, sb.color_palette("colorblind", len(conditions))))

def plot_transition_readouts(drug, df_t, n_types=10):
    """Post - pre change of each readout for the most frequent transition types,
    mean +/- sem over animals (one value per animal), by condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cond_colors = condition_colors()
    types =(df_t.dropna(subset=[f"d_{readouts[0]}"]).groupby("transition")["mouse"].nunique()
             .sort_values(ascending=False).index[:n_types].tolist())
    if not types:
        return
    conds = [c for c in conditions if c in df_t["condition"].unique()]
    x = np.arange(len(types))
    w = 0.8 / max(len(conds), 1)
    fig, axes = plt.subplots(len(readouts), 1, figsize=(max(10, 1.4 * len(types) + 2), 13),
                             sharex=True, layout="tight")
    for ax, q in zip(axes, [f"d_{q}" for q in readouts]):
        pa = df_t.dropna(subset=[q]).groupby(["transition", "condition", "mouse"])[q].mean()
        g = pa.groupby(level=["transition", "condition"]).agg(["mean", "sem", "count"])
        g = g[g["count"] >= min_sessions_per_cell]          # same animal threshold as the stats
        for i, c in enumerate(conds):
            m = [g["mean"].get((t, c), np.nan) for t in types]
            e = [g["sem"].get((t, c), np.nan) for t in types]
            ax.bar(x + (i - (len(conds) - 1) / 2) * w, m, w, yerr=e, capsize=2,
                   color=cond_colors[c], label=c)
        ax.axhline(0, c="0.5", lw=1)
        ax.set(ylabel=f"{q[2:]}: post - pre")
    axes[0].legend(fontsize=9)
    axes[-1].set_xticks(x, [f"S{t.replace('->', '→S')}" for t in types])
    axes[-1].set_xlabel("transition (speed-ordered states)")
    fig.suptitle(f"SPN readout change across state transitions - {drug} "
                 f"(±{trans_halfwin / fs:.0f} s windows)", fontsize=13)
    fig.savefig(f"{out_dir}/{out_prefix}_spn_transition_readouts_{drug}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
