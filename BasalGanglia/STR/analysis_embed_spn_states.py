"""Embedding-based behavioural states x SPN firing-rate dynamics.

Alternative to analysis_arhmm_spn_states.py: instead of a generative AR-HMM,
behaviour states are found by

  1. delay-embedding the [speed, acceleration] traces (each frame -> a vector of
     the current + past taps; as many taps as the AR-HMM's AR order),
  2. clustering the pooled delay-embedded features directly (k-means / GMM) into
     behaviour states, on a subsample,
  3. propagating the cluster labels to *every* frame with a kNN classifier in the
     delay-embedded feature space; frames with a low kNN vote fraction are left
     UNSPECIFIED (-1), analogous to the HMM's posterior cutoff,
  4. a 2-D t-SNE / Isomap embedding of the same subsample, used ONLY to visualise
     the clusters (it is not clustered on).

Everything downstream of the state labels is IDENTICAL to the HMM pipeline -- it
calls A.analyze_states (same sessions, drugs, conditions, D1/D2 split, readouts,
transition readouts, statistics, tables and figures). Outputs use an `embed_`
prefix in their own folder.

Run with the dedicated env:
  /home/rgast/conda/envs/arhmm/bin/python analysis_embed_spn_states.py
"""

import os
import pickle
import numpy as np
import analysis_arhmm_spn_states as A   # reuse loading, readouts, stats, transitions

# ---------------------------------------------------------------------------
# configuration (data / readout params are inherited from A; below is embedding)
# ---------------------------------------------------------------------------

out_dir = f"{A.save_dir}/embed_all10/K{A.K_states}"
seed = A.seed
plot_results = A.plot_results
model_path = f"{out_dir}/embed_model.pkl"

# load an existing embed_model.pkl and redo only the post-clustering analysis
# (e.g. after changing min_cluster_confidence). Not overwritten in this mode.
load_model = False

# delay embedding
embed_lags = A.ar_lags   # number of taps (current + past); feature dim = 2*embed_lags
embed_step = 1           # spacing between taps (frames)

# 2-D embedding (VISUALISATION ONLY; clustering is done in feature space)
embed_method = "tsne"    # "tsne" | "isomap"
n_embed = 20000          # frames subsampled (pooled) for clustering + embedding
tsne_perplexity = 5
isomap_neighbors = 10

# clustering in the ORIGINAL delay-embedding feature space (not the 2-D embedding)
n_clusters = A.K_states  # number of behaviour states (same K as the AR-HMM)
cluster_method = "gmm"   # "kmeans" | "gmm"

# label propagation to all frames
knn_k = 15               # neighbours for the kNN classifier (feature space)
# frames whose kNN vote fraction for the winning cluster is below this are left
# UNSPECIFIED (-1), analogous to the HMM's min_state_posterior (0 -> disabled)
min_cluster_confidence = A.min_state_posterior

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
    n_unspec = n_tot = 0
    for s in sessions:
        F = s["feat"]
        if min_cluster_confidence > 0:
            proba = knn.predict_proba(F)
            raw = knn.classes_[proba.argmax(1)]
            raw = np.where(proba.max(1) >= min_cluster_confidence, raw, -1)
            n_unspec += int(np.sum(raw < 0)); n_tot += len(raw)
        else:
            raw = knn.predict(F)
        s["z"] = raw.astype(int)
    if min_cluster_confidence > 0:
        print(f"Unspecified (kNN vote fraction < {min_cluster_confidence}): "
              f"{n_unspec}/{n_tot} frames ({100 * n_unspec / max(n_tot, 1):.1f}%)")

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
                 emb2d=emb, emb_labels=lab, drugs=A.drugs, conditions=A.conditions)
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
    fig.savefig(f"{out_dir}/embed_2d_embedding.png", dpi=130)
    plt.close(fig)
    print(f"Saved 2-D embedding scatter to {out_dir}/embed_2d_embedding.png")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
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

    # identical downstream analysis to the AR-HMM pipeline, written to out_dir
    A.out_dir, A.out_prefix, A.method_name = out_dir, "embed", "Embedding"
    A.analyze_states(sessions, state_speed, len(state_speed), None, rng)


if __name__ == "__main__":
    main()
