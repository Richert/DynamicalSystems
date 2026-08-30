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
  5. Per (session, state): pool all state frames and compute the firing-rate CDF
     centre & slope and the SPN dimensionality (participation ratio), subsampling
     to a common neuron count and time count so the estimates are comparable.
  6. Transition-aligned variant: peri-transition SPN population-rate traces.
  7. Mixed-effects statistics (value ~ state*condition + (1|animal)) + occupancy.

Run with the dedicated env (has ssm + statsmodels):
  /home/rgast/conda/envs/arhmm/bin/python analysis_arhmm_spn_states.py
"""

import os
import pickle
import numpy as np
import h5py
from scipy.special import erf

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

neural_path = "/home/rgast/data/parker_data/neural_data"
h5_path = "/home/rgast/data/parker_data/oasis_rates.h5"
save_dir = "/home/rgast/data/parker_data"
speed_field = "speed_traces_5hz"
fs = 5.0

# which drugs / conditions to include
drugs = ["haloperidol"]
# map (dose, raw condition) -> analysis label; None -> drop the session
condition_map = {
    ("Vehicle", "veh"): "Control",
    ("Vehicle", "amph"): "Amphetamine",
    ("HighDose", "amph"): "A + Drug",
}
conditions = ["Control", "Amphetamine", "A + Drug"]

# AR-HMM
# load the existing arhmm_model.pkl instead of training, and only redo the
# post-training analysis (labelling/readouts/stats/plots) -- e.g. after changing
# min_state_posterior below. The saved model is not overwritten in this mode.
load_model = False
K_states = 5              # number of HMM states (set select_K=True to sweep)
ar_lags = 5              # AR order (temporal context; ~0.6 s at 5 Hz)
kappa = 100.0             # stickiness (self-transition prior); larger -> more persistent
em_iters = 50
n_restarts = 10           # EM restarts; keep the highest-likelihood fit
fit_cap_frames = 4500    # per-session cap for the *fit* (balances long amph sessions)
select_K = True         # if True, sweep K_grid to pick K
K_grid = [3, 4, 5, 6]
k_criterion = "bic"      # model selection: "bic" | "aic" (penalise params) | "heldout" (LL/frame)
k_sweep_jobs = None      # parallel processes for the K sweep (None -> min(len(K_grid), cpu_count); 1 -> serial)
# frames whose most-likely state has a posterior below this cutoff are left
# UNSPECIFIED (state id -1) instead of being forced into a state (0 -> disabled)
min_state_posterior = 0.6
seed = 1

# SPN readouts
rate_lims = (1e-6, 1e-3)     # winsorisation for the log-rate CDF (as in the figures)
n_neurons_common = 30        # subsample neurons to this many (comparable dimensionality)
T_min = 40                   # min frames of a state in a session to compute CDF centre/slope
T_common = 60                # frames used for dimensionality (subsample); needs >= this many

# animal-specific behaviour<->neural lag
max_lag_frames = 10          # search +/- this many frames (2 s at 5 Hz)

# transition-aligned analysis
trans_halfwin = 15           # +/- frames around a transition (3 s)
min_dwell = 5                # min dwell (frames) on both sides to count a transition

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

def load_sessions():
    """Load OASIS rates (h5) + speed (mat), aligned per session."""
    sessions = []
    with h5py.File(h5_path, "r") as h5:
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
            mat = f"{neural_path}/{drug}/{dose}/{session}/{cond_raw}_drug.mat"
            if not os.path.isfile(mat):
                continue
            try:
                from scipy.io import loadmat
                speed = np.asarray(loadmat(mat, simplify_cells=True)[f"{cond_raw}_drug"][speed_field],
                                   dtype=float)
            except Exception:
                continue
            rates = h5[name][...].astype(float)                  # (N, T)
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

def fit_arhmm(datas, K, rng):
    """Fit a sticky AR-HMM with several EM restarts; keep the best-likelihood fit."""
    import ssm
    import numpy.random as npr
    best, best_ll = None, -np.inf
    for r in range(n_restarts):
        npr.seed(int(rng.integers(0, 2 ** 31)))
        hmm = ssm.HMM(K, 2, observations="ar", transitions="sticky",
                      observation_kwargs=dict(lags=ar_lags), transition_kwargs=dict(kappa=kappa))
        hmm.fit(datas, method="em", num_iters=em_iters, verbose=0)
        ll = hmm.log_likelihood(datas)
        if ll > best_ll:
            best, best_ll = hmm, ll
    return best

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

# module-level worker state/functions so the K-sweep can run in a process pool
_SWEEP = {}

def _sweep_init(fit_data, eval_data):
    _SWEEP["fit"], _SWEEP["eval"] = fit_data, eval_data

def _fit_one_K(k_seed):
    """Fit one K on the shared fit data; return (K, logL on eval data, n_eval,
    K_eff, fitted hmm). K_eff = number of states actually used (non-zero Viterbi
    occupancy) -- a fit that collapses to fewer states is scored at its effective
    K, so dead states cost nothing and don't inflate the selected K. The hmm is
    returned so the selected model can be reused without an extra training run."""
    K, s = k_seed
    hmm = fit_arhmm(_SWEEP["fit"], K, np.random.default_rng(s))
    used = set()
    for d in _SWEEP["eval"]:
        used.update(np.unique(hmm.most_likely_states(np.ascontiguousarray(d))).tolist())
    return (K, float(hmm.log_likelihood(_SWEEP["eval"])),
            int(sum(len(d) for d in _SWEEP["eval"])), int(len(used)), hmm)

def _n_params(K):
    """Free parameters of the sticky AR-HMM: transitions + initial + per-state
    AR-Gaussian observations (D=2 obs dim, ar_lags AR order)."""
    D = 2
    per_state = D * D * ar_lags + D + D * (D + 1) // 2   # AR matrix + bias + covariance
    return K * (K - 1) + (K - 1) + K * per_state          # transitions + initial + obs

def choose_K(sessions, rng):
    """Sweep K_grid and pick K by BIC/AIC (penalise parameters, so dead states
    don't win) or held-out LL/frame. Each K is an independent fit, parallelised."""
    if k_criterion == "heldout":
        idx = rng.permutation(len(sessions))
        n_test = max(1, len(sessions) // 5)
        test, train = idx[:n_test], idx[n_test:]
        fit_data = fit_datasets([sessions[i] for i in train], rng)
        eval_data = [np.ascontiguousarray(sessions[i]["obs_z"]) for i in test]
    else:                                                # AIC/BIC: fit and score on all data
        fit_data = eval_data = fit_datasets(sessions, rng)
    tasks = [(K, int(rng.integers(0, 2 ** 31))) for K in K_grid]   # a seed per K

    n_jobs = min(len(K_grid), k_sweep_jobs or (os.cpu_count() or 1))
    if n_jobs <= 1 or len(K_grid) <= 1:
        _sweep_init(fit_data, eval_data)
        res = [_fit_one_K(t) for t in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor
        print(f"  sweeping K={K_grid} over {n_jobs} parallel processes ({k_criterion}) ...")
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_sweep_init,
                                 initargs=(fit_data, eval_data)) as ex:
            res = list(ex.map(_fit_one_K, tasks))

    ll = {K: L for K, L, _, _, _ in res}
    n = {K: nn for K, _, nn, _, _ in res}
    keff = {K: ke for K, _, _, ke, _ in res}     # effective K (dead states dropped)
    models = {K: h for K, _, _, _, h in res}
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

def cdf_center_slope(mean_rates):
    x = to_lograte(mean_rates)
    return float(np.mean(x)), float(1.0 / (np.std(x) + 1e-12))   # center, slope=1/sigma

def participation_ratio(R):
    """PR of the neuron covariance of R (n_neurons x n_time)."""
    C = np.cov(R, ddof=0)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 0]
    return float(np.sum(ev) ** 2 / (np.sum(ev ** 2) * 1.0)) if len(ev) else np.nan

def state_readouts(sessions, lag_of, rng):
    """One (center, slope, dim, occupancy) per (session, state), lag-corrected."""
    rows = []
    for s in sessions:
        N = s["rates"].shape[0]
        lag = lag_of.get(s["mouse"], 0)
        for k in np.unique(s["z"]):
            if k < 0:                      # skip UNSPECIFIED frames (low-posterior)
                continue
            frames = np.where(s["z"] == k)[0]
            nf = len(frames)
            neu_frames = frames - lag
            valid = (neu_frames >= 0) & (neu_frames < s["rates"].shape[1])
            neu_frames = neu_frames[valid]
            row = dict(mouse=s["mouse"], condition=s["condition"], spn=s["spn"],
                       drug=s["drug"], session=s["session"], state=int(k),
                       occupancy=nf / len(s["z"]), n_frames=nf,
                       center=np.nan, slope=np.nan, dim=np.nan)
            if len(neu_frames) >= T_min and N >= n_neurons_common:
                neu = rng.choice(N, n_neurons_common, replace=False)
                R = s["rates"][np.ix_(neu, neu_frames)]
                row["center"], row["slope"] = cdf_center_slope(R.mean(axis=1))
                if len(neu_frames) >= T_common:
                    cols = rng.choice(len(neu_frames), T_common, replace=False)
                    row["dim"] = participation_ratio(R[:, cols])
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
        chg = np.where(np.diff(z) != 0)[0] + 1       # transition onset frames
        for t in chg:
            frm, to = int(z[t - 1]), int(z[t])
            # require stable dwell on both sides
            if t - min_dwell < 0 or t + min_dwell > len(z):
                continue
            if not (np.all(z[t - min_dwell:t] == frm) and np.all(z[t:t + min_dwell] == to)):
                continue
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

def run_stats(df):
    """Mixed-effects test per quantity. Keeps only (state,condition) cells with
    enough animals, then tries interaction -> additive -> OLS, falling back on
    singular designs (empty interaction cells are common when a state is rare in
    a condition). Returns the p-values of whichever model fitted, FDR-corrected."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests

    def fit_one(q):
        d = df.dropna(subset=[q]).copy()
        cell = d.groupby(["state", "condition"])["mouse"].nunique()
        good = cell[cell >= min_sessions_per_cell].index
        d = d[d.set_index(["state", "condition"]).index.isin(good)].copy()
        # keep only states/conditions still present in >=2 of the other factor
        for _ in range(3):
            sc = d.groupby("state")["condition"].nunique()
            d = d[d["state"].isin(sc[sc >= 2].index)]
            cs = d.groupby("condition")["state"].nunique()
            d = d[d["condition"].isin(cs[cs >= 2].index)]
        if d["state"].nunique() < 2 or d["condition"].nunique() < 2 or len(d) < 8:
            return None
        d["state"] = d["state"].astype("category")
        for formula, kind in [(f"{q} ~ C(state)*C(condition)", "mixedlm-interaction"),
                              (f"{q} ~ C(state)+C(condition)", "mixedlm-additive")]:
            try:
                mdf = smf.mixedlm(formula, d, groups=d["mouse"]).fit(reml=False, method="lbfgs")
                if not np.all(np.isfinite(mdf.pvalues.values)):
                    raise ValueError("non-finite p-values")
                return d, mdf, kind
            except Exception:
                continue
        # last resort: OLS with mouse-clustered robust SEs (guards pseudoreplication)
        try:
            mdf = smf.ols(f"{q} ~ C(state)+C(condition)", d).fit(
                cov_type="cluster", cov_kwds={"groups": d["mouse"]})
            return d, mdf, "ols-clustered-additive"
        except Exception as e:
            return ("error", str(e))

    results = {}
    for q in ["center", "slope", "dim"]:
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
            results[q] = dict(model=mdf, kind=kind,
                              fdr={k: (float(p), float(qa), bool(rj))
                                   for k, p, qa, rj in zip(pvals.index, pvals.values, q_adj, rej)})
    return results

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    from pandas import DataFrame
    rng = np.random.default_rng(seed)

    sessions = load_sessions()

    if load_model:
        # reuse the existing fit; redo only the post-training analysis
        m = pickle.load(open(f"{save_dir}/arhmm_model.pkl", "rb"))
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
        pickle.dump(dict(hmm=hmm, mu=mu, sd=sd, K=K, ar_lags=ar_lags, kappa=kappa),
                    open(f"{save_dir}/arhmm_model.pkl", "wb"))
        print(f"Saved fitted AR-HMM to {save_dir}/arhmm_model.pkl")

    state_speed = relabel_by_speed(sessions, hmm, K)     # applies min_state_posterior cutoff
    print("State mean speed (cm/s), ascending:",
          np.array2string(state_speed, precision=2))

    lag_of = animal_lags(sessions)
    print("Animal lags (frames):", {k: lag_of[k] for k in list(lag_of)[:8]}, "...")

    rows = state_readouts(sessions, lag_of, rng)
    df = DataFrame(rows)
    df.to_csv(f"{save_dir}/arhmm_spn_state_readouts.csv", index=False)  # drug-labelled

    # --- per-drug analysis, using the SHARED state labels ---
    per_drug = {}
    for drug in [d for d in drugs if d in set(df["drug"])]:
        print(f"\n########################  drug = {drug}  ########################")
        df_d = df[df["drug"] == drug]
        sess_d = [s for s in sessions if s["drug"] == drug]
        lag_d = {m: lag_of[m] for m in df_d["mouse"].unique() if m in lag_of}

        occ = df_d.groupby(["state", "condition"])["occupancy"].mean().unstack("condition")
        print("Mean occupancy by state x condition:\n", occ.round(3))

        print("\n=== mixed-effects statistics (value ~ state*condition + (1|animal)) ===")
        stats = run_stats(df_d)
        for q, r in stats.items():
            print(f"\n--- {q} ---")
            if r is None:
                print("  insufficient data")
            elif "error" in r:
                print("  model failed:", r["error"])
            else:
                print(f"  [{r['kind']}]")
                for term, (p, qa, sig) in r["fdr"].items():
                    print(f"  {term:32s} p={p:.4g}  q(FDR)={qa:.4g} {'*' if sig else ''}")

        trans = transition_traces(sess_d, lag_of)
        # store only picklable summaries (drop fitted model objects)
        per_drug[drug] = dict(
            occupancy=occ, transitions=trans, lag_of=lag_d,
            stats={q: (r if (r is None or "error" in r)
                       else dict(kind=r["kind"], fdr=r["fdr"])) for q, r in stats.items()})
        if plot_results:
            make_plots(drug, df_d, state_speed, trans, occ, lag_d)

    # persist shared model summary + per-drug results (input for the cross-drug script)
    pickle.dump(dict(state_speed=state_speed, K=K, kscores=kscores, lag_of=lag_of,
                     ar_lags=ar_lags, drugs=list(per_drug), per_drug=per_drug),
                open(f"{save_dir}/arhmm_spn_results.pkl", "wb"))
    print(f"\nSaved readouts CSV and per-drug results pickle to {save_dir}/")

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
    cond_colors = {"Control": sb.color_palette("colorblind")[0],
                   "Amphetamine": sb.color_palette("colorblind")[1],
                   "A + Drug": sb.color_palette("colorblind")[2]}

    # (1) state signatures (mean speed) + occupancy; (2) readouts; (3) lags
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="tight")
    fig.suptitle(f"AR-HMM behavioural states x SPN dynamics - {drug}", fontsize=15)

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

    for ax, q in zip(axes[1], ["center", "slope", "dim"]):
        g = df.dropna(subset=[q]).groupby(["state", "condition"])[q].agg(["mean", "sem"])
        for c in [c for c in conditions if c in df["condition"].unique()]:
            sub = g.xs(c, level="condition") if c in g.index.get_level_values("condition") else None
            if sub is None:
                continue
            ax.errorbar(sub.index, sub["mean"], yerr=sub["sem"], marker="o",
                        color=cond_colors[c], label=c, capsize=3)
        ax.set(xlabel="state", ylabel=q, title=f"SPN {q} by state x condition")
    axes[1, 0].legend(fontsize=9)
    fig.savefig(f"{save_dir}/arhmm_spn_states_summary_{drug}.png", dpi=120)

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
        fig2.savefig(f"{save_dir}/arhmm_spn_transitions_{drug}.png", dpi=120)
    print(f"Saved figures to {save_dir}/arhmm_spn_*_{drug}.png")


if __name__ == "__main__":
    main()
