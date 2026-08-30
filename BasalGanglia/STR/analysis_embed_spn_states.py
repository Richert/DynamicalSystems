"""Embedding-based behavioural states x SPN firing-rate dynamics.

Alternative to analysis_arhmm_spn_states.py: instead of a generative AR-HMM,
behaviour states are found by

  1. delay-embedding the [speed, acceleration] traces (each frame -> a vector of
     the current + past taps),
  2. clustering the pooled delay-embedded features directly (k-means / GMM) into
     behaviour states, on a subsample,
  3. propagating the cluster labels to *every* frame with a kNN classifier in the
     delay-embedded feature space,
  4. a 2-D t-SNE / Isomap embedding of the same subsample, used ONLY to visualise
     the clusters (it is not clustered on).

Everything downstream is IDENTICAL to the HMM script (it reuses those functions):
each frame gets a state label, then per (session, state) firing-rate CDF
centre/slope + SPN dimensionality, animal-specific lag, transition-aligned
traces, occupancy, and mixed-effects stats. Outputs use an `embed_` prefix.

Run with the dedicated env:
  /home/rgast/conda/envs/arhmm/bin/python analysis_embed_spn_states.py
"""

import pickle
import numpy as np
import analysis_arhmm_spn_states as A   # reuse loading, readouts, stats, transitions

# ---------------------------------------------------------------------------
# configuration (data / readout params are inherited from A; below is embedding)
# ---------------------------------------------------------------------------

save_dir = A.save_dir
drugs = A.drugs
conditions = A.conditions
fs = A.fs
seed = A.seed
plot_results = A.plot_results
model_path = f"{save_dir}/embed_model.pkl"

# load an existing embed_model.pkl and redo only the post-clustering analysis
# (e.g. after changing min_cluster_confidence). Not overwritten in this mode.
load_model = False

# delay embedding
embed_lags = 5           # number of taps (current + past); feature dim = 2*embed_lags
embed_step = 1           # spacing between taps (frames)

# 2-D embedding (VISUALISATION ONLY; clustering is done in feature space)
embed_method = "tsne"    # "tsne" | "isomap"
n_embed = 20000          # frames subsampled (pooled) for the embedding
tsne_perplexity = 5
isomap_neighbors = 10

# clustering in the ORIGINAL delay-embedding feature space (not the 2-D embedding)
n_clusters = 5           # number of behaviour states
cluster_method = "gmm"   # "kmeans" | "gmm"

# label propagation to all frames
knn_k = 15               # neighbours for the kNN classifier (feature space)
# frames whose kNN vote fraction for the winning cluster is below this are left
# UNSPECIFIED (-1), analogous to the HMM's min_state_posterior (0 -> disabled)
min_cluster_confidence = 0.0

# ---------------------------------------------------------------------------
# feature construction + clustering
# ---------------------------------------------------------------------------

def build_features(s):
    """Delay-embed the standardised [speed, accel] of one session -> (T, 2*lags),
    padding the start by repeating the first frame."""
    z = s["obs_z"]                                   # (T, 2) standardised speed+accel
    T = len(z)
    idx = np.clip(np.arange(T)[:, None] - np.arange(embed_lags)[None, :] * embed_step, 0, T - 1)
    return z[idx].reshape(T, -1)                     # (T, 2*embed_lags)

def predict_raw(sessions, knn):
    """Assign each frame its raw cluster id (or -1 below the confidence cutoff)."""
    for s in sessions:
        F = s["feat"]
        if min_cluster_confidence > 0:
            proba = knn.predict_proba(F)
            raw = knn.classes_[proba.argmax(1)]
            raw = np.where(proba.max(1) >= min_cluster_confidence, raw, -1)
        else:
            raw = knn.predict(F)
        s["z"] = raw.astype(int)

def relabel_clusters_by_speed(sessions, n_clusters):
    """Drop empty clusters and renumber survivors 0..K_eff-1 by ascending mean
    speed. Returns (mean_speed_sorted, remap: raw_id -> new_id)."""
    sums = np.zeros(n_clusters); cnts = np.zeros(n_clusters)
    for s in sessions:
        z = s["z"]
        for k in range(n_clusters):
            m = z == k
            sums[k] += s["speed"][m].sum(); cnts[k] += m.sum()
    used = np.where(cnts > 0)[0]
    if len(used) < n_clusters:
        print(f"Dropping {n_clusters - len(used)} empty cluster(s); effective K = {len(used)}")
    mean_speed = sums[used] / cnts[used]
    order = np.argsort(mean_speed)
    remap = {int(used[o]): i for i, o in enumerate(order)}
    for s in sessions:
        z = s["z"]; znew = np.full_like(z, -1)
        for old, new in remap.items():
            znew[z == old] = new
        s["z"] = znew
    return mean_speed[order], remap

def embed_and_cluster(sessions, rng):
    """Delay-embed -> cluster in the feature space -> kNN propagate to all frames.
    A 2-D t-SNE/Isomap embedding of the same subsample is computed purely for
    visualisation (it is NOT clustered on)."""
    from sklearn.manifold import TSNE, Isomap
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import KNeighborsClassifier

    for s in sessions:
        s["feat"] = build_features(s)
    allfeat = np.concatenate([s["feat"] for s in sessions], axis=0)
    N = len(allfeat)
    ne = min(n_embed, N)
    sub = rng.choice(N, ne, replace=False)
    Xsub = allfeat[sub]

    # --- clustering: in the ORIGINAL delay-embedding feature space ---
    print(f"Clustering {ne}/{N} frames into {n_clusters} states via {cluster_method} "
          f"in the {Xsub.shape[1]}-D delay-embedding feature space ...")
    if cluster_method == "gmm":
        lab = GaussianMixture(n_clusters, random_state=seed).fit_predict(Xsub)
    else:
        lab = KMeans(n_clusters, random_state=seed, n_init=10).fit_predict(Xsub)

    # --- 2-D embedding: visualisation only (coloured by the feature-space labels) ---
    print(f"Embedding the same frames into 2-D via {embed_method} (visualisation only) ...")
    if embed_method == "isomap":
        emb = Isomap(n_neighbors=isomap_neighbors, n_components=2).fit_transform(Xsub)
    else:
        emb = TSNE(n_components=2, perplexity=tsne_perplexity, init="pca",
                   random_state=seed).fit_transform(Xsub)

    knn = KNeighborsClassifier(n_neighbors=knn_k).fit(Xsub, lab)
    predict_raw(sessions, knn)
    state_speed, remap = relabel_clusters_by_speed(sessions, n_clusters)

    model = dict(knn=knn, remap=remap, n_clusters=n_clusters, state_speed=state_speed,
                 embed_method=embed_method, embed_lags=embed_lags, embed_step=embed_step,
                 emb2d=emb, emb_labels=lab)
    return state_speed, model, emb, lab, remap

# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_embedding(emb, labels, remap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sb
    new = np.array([remap.get(int(l), -1) for l in labels])      # speed-ordered ids
    K = len(set(v for v in new if v >= 0))
    colors = sb.color_palette("Set2", K)
    fig, ax = plt.subplots(figsize=(6, 5.5), layout="tight")
    for k in range(K):
        m = new == k
        ax.scatter(emb[m, 0], emb[m, 1], s=4, color=colors[k], alpha=0.5, label=f"S{k}")
    ax.set(xlabel=f"{embed_method}-1", ylabel=f"{embed_method}-2",
           title=f"2-D behaviour embedding ({embed_method}), {K} clusters")
    ax.legend(markerscale=3, fontsize=9)
    fig.savefig(f"{save_dir}/embed_2d_embedding.png", dpi=130)
    print(f"Saved 2-D embedding scatter to {save_dir}/embed_2d_embedding.png")

def make_plots(drug, df, state_speed, trans, occ, lag_of):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sb
    sb.set_palette("colorblind")
    K = len(state_speed)
    cc = {"Control": sb.color_palette("colorblind")[0],
          "Amphetamine": sb.color_palette("colorblind")[1],
          "A + Drug": sb.color_palette("colorblind")[2]}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="tight")
    fig.suptitle(f"Embedding behavioural states x SPN dynamics - {drug}", fontsize=15)
    axes[0, 0].bar(range(K), state_speed, color="0.5")
    axes[0, 0].set(xlabel="state (speed-ordered)", ylabel="mean speed (cm/s)",
                   title="state signatures (embedding clusters)")
    occ_p = occ.reindex(columns=[c for c in conditions if c in occ.columns])
    occ_p.plot(kind="bar", ax=axes[0, 1], color=[cc[c] for c in occ_p.columns])
    axes[0, 1].set(xlabel="state", ylabel="occupancy", title="occupancy by condition")
    axes[0, 2].hist(list(lag_of.values()), bins=range(-A.max_lag_frames, A.max_lag_frames + 2),
                    color="0.5")
    axes[0, 2].set(xlabel="lag (frames)", ylabel="# animals", title="animal neural<->behaviour lag")
    for ax, q in zip(axes[1], ["center", "slope", "dim"]):
        g = df.dropna(subset=[q]).groupby(["state", "condition"])[q].agg(["mean", "sem"])
        for c in [c for c in conditions if c in df["condition"].unique()]:
            if c in g.index.get_level_values("condition"):
                sub = g.xs(c, level="condition")
                ax.errorbar(sub.index, sub["mean"], yerr=sub["sem"], marker="o",
                            color=cc[c], label=c, capsize=3)
        ax.set(xlabel="state", ylabel=q, title=f"SPN {q} by state x condition")
    axes[1, 0].legend(fontsize=9)
    fig.savefig(f"{save_dir}/embed_spn_states_summary_{drug}.png", dpi=120)

    keys = sorted(trans, key=lambda k: -trans[k]["n_trans"])[:6]
    if keys:
        t = (np.arange(-A.trans_halfwin, A.trans_halfwin + 1)) / fs
        fig2, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.4),
                                 sharey=True, layout="tight")
        axs = np.atleast_1d(axs)
        for ax, key in zip(axs, keys):
            frm, to, cond = key; d = trans[key]
            ax.plot(t, d["mean"], color=cc.get(cond, "k"))
            ax.fill_between(t, d["mean"] - d["sem"], d["mean"] + d["sem"], alpha=0.3, color=cc.get(cond, "k"))
            ax.axvline(0, ls="--", c="0.5", lw=1)
            ax.set(title=f"S{frm}->S{to}\n{cond} (n={d['n_mice']}m)", xlabel="time from transition (s)")
        axs[0].set_ylabel("pop. rate (z)")
        fig2.suptitle(f"transition-aligned SPN population rate - {drug}", fontsize=13)
        fig2.savefig(f"{save_dir}/embed_spn_transitions_{drug}.png", dpi=120)
    print(f"Saved figures to {save_dir}/embed_spn_*_{drug}.png")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    from pandas import DataFrame
    rng = np.random.default_rng(seed)
    sessions = A.load_sessions()

    if load_model:
        m = pickle.load(open(model_path, "rb"))
        mu, sd, remap = m["mu"], m["sd"], m["remap"]
        for s in sessions:
            s["obs_z"] = (s["obs"] - mu) / sd
            s["feat"] = build_features(s)
        predict_raw(sessions, m["knn"])                  # raw cluster ids (with -1 low-conf)
        lut = np.full(m["n_clusters"], -1, int)
        for old, new in remap.items():
            lut[old] = new
        for s in sessions:
            v = s["z"] >= 0
            z = np.full_like(s["z"], -1); z[v] = lut[s["z"][v]]; s["z"] = z
        state_speed = m["state_speed"]
        print(f"Loaded existing embedding model (K={len(state_speed)}); redoing analysis only.")
    else:
        mu, sd = A.standardize(sessions)
        state_speed, model, emb, lab, remap = embed_and_cluster(sessions, rng)
        model.update(mu=mu, sd=sd)
        pickle.dump(model, open(model_path, "wb"))
        print(f"Saved embedding model to {model_path}")
        if plot_results:
            plot_embedding(emb, lab, remap)

    print("State mean speed (cm/s), ascending:", np.array2string(state_speed, precision=2))

    lag_of = A.animal_lags(sessions)
    rows = A.state_readouts(sessions, lag_of, rng)
    df = DataFrame(rows)
    df.to_csv(f"{save_dir}/embed_spn_state_readouts.csv", index=False)

    per_drug = {}
    for drug in [d for d in drugs if d in set(df["drug"])]:
        print(f"\n########################  drug = {drug}  ########################")
        df_d = df[df["drug"] == drug]
        sess_d = [s for s in sessions if s["drug"] == drug]
        lag_d = {mo: lag_of[mo] for mo in df_d["mouse"].unique() if mo in lag_of}
        occ = df_d.groupby(["state", "condition"])["occupancy"].mean().unstack("condition")
        print("Mean occupancy by state x condition:\n", occ.round(3))
        print("\n=== mixed-effects statistics (value ~ state*condition + (1|animal)) ===")
        stats = A.run_stats(df_d)
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
        trans = A.transition_traces(sess_d, lag_of)
        per_drug[drug] = dict(occupancy=occ, transitions=trans, lag_of=lag_d,
                              stats={q: (r if (r is None or "error" in r)
                                         else dict(kind=r["kind"], fdr=r["fdr"])) for q, r in stats.items()})
        if plot_results:
            make_plots(drug, df_d, state_speed, trans, occ, lag_d)

    pickle.dump(dict(state_speed=state_speed, K=len(state_speed), lag_of=lag_of,
                     embed_method=embed_method, n_clusters=n_clusters,
                     drugs=list(per_drug), per_drug=per_drug),
                open(f"{save_dir}/embed_spn_results.pkl", "wb"))
    print(f"\nSaved readouts CSV and per-drug results pickle to {save_dir}/")


if __name__ == "__main__":
    main()
