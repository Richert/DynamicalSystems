"""Decoding behaviour from SPN activity, per recording: discrete behaviour labels (multinomial
logistic regression) vs continuous pose features (ridge regression), each from spikes and from
denoised traces, with two-stage cross-checks that put both on a common footing.

Neural input (per draw: n_sub cells, z-scored): spikes smoothed with a Gaussian kernel
(kernel_s) or the denoised trace C / noise; each cell at the offsets lags_s around the
behaviour frame (window covers the spike lead and the trace lag of analysis_spn_behavior_lags).
Behaviour (data file, fixed video offset): label_clean (artefacts / short bouts excluded) and
the 11 AR-HMM features in the HMM's transforms. Fits on every step-th frame (5 Hz).

Each recording is decoded in seg_s segments (15 min: Control 1, Amphetamine 3; averaged), so
all decoders train on the same amount of data.
Cross-validation: n_folds contiguous blocks, gap_s left out on both sides of each test block;
hyper-parameters (C, alpha) chosen on the last inner block of the training data.
Per recording, readout and draw:
  labels    balanced accuracy of neural -> label (direct), of neural -> decoded features ->
            label (via features: out-of-fold feature predictions, then logistic), and of the
            true features -> label (ceiling: how well the pose features explain the labels);
            circular-shift null of the direct decoder
  features  Pearson r within test blocks (see corr; and R^2) per feature of neural -> feature (direct), of neural -> decoded label
            probabilities -> feature (via labels), and of true labels -> feature (ceiling);
            circular-shift null
direct > via features: label information beyond the (linearly decodable) pose features;
direct > via labels: continuous pose information beyond the labels.

Outputs (OUT_DIR): decoding.csv (one row per recording x readout x draw x target), figures.

  conda run -n parker python analysis_spn_decoding.py [n_jobs]
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter1d

dataset = os.environ.get("DATASET", "/home/rgast/data/parker_data/dataset/clozapine_spn.h5")
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/spn_decoding")
conditions = ["Control", "Amphetamine", "A + Clozapine high"]
fs = 20.0
kernel_s = 0.1
lags_s = (-0.4, -0.2, 0.0, 0.2, 0.4)
step = 4                                    # fit at 5 Hz
n_sub, n_draw = 50, 3
n_folds, gap_s = 5, 2.0
seg_s = 900.0                               # decode in segments of this length (equal training data)
min_class_s = 10.0                          # label used with >= this much time
null_min_s = 60.0
Cs = (0.01, 0.1, 1.0)
alphas = (10.0, 100.0, 1e3, 1e4, 1e5)
speed_eps, turn_eps = 0.5, 5.0
features = ["log_speed", "turn_asinh", "head_angle", "body_len", "bend", "log_head_speed",
            "head_turn_asinh", "ear_width", "hip_width", "wall_dist", "wall_ahead"]
seed = 1

def behaviour(b):
    raw = {c: b[c][:] for c in ("speed", "turn", "head_angle", "body_len", "bend", "head_speed",
                                "head_turn", "ear_width", "hip_width", "wall_dist", "wall_ahead")}
    F = np.column_stack([np.log10(raw["speed"] + speed_eps), np.arcsinh(raw["turn"] / turn_eps),
                         raw["head_angle"], raw["body_len"], raw["bend"],
                         np.log10(raw["head_speed"] + speed_eps),
                         np.arcsinh(raw["head_turn"] / turn_eps), raw["ear_width"],
                         raw["hip_width"], raw["wall_dist"], raw["wall_ahead"]])
    return F, b["label_clean"][:], list(b.attrs["label_names"])

def lagged(X, idx):
    """Rows idx of X (cells x frames, z-scored) at the offsets lags_s -> (len(idx), cells x lags)."""
    T = X.shape[1]
    cols = [X[:, np.clip(idx + int(round(l * fs)), 0, T - 1)].T for l in lags_s]
    return np.hstack(cols)

def folds(n, n_gap):
    """(train, test) index arrays of contiguous blocks with gaps."""
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        test = np.arange(a, b)
        train = np.r_[np.arange(0, max(0, a - n_gap)), np.arange(min(n, b + n_gap), n)]
        out.append((train, test))
    return out

def zs(Xtr, Xte):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - m) / s, (Xte - m) / s

def bal_acc(y, p, classes):
    return np.mean([np.mean(p[y == c] == c) for c in classes if (y == c).any()])

def fit_logistic(Xtr, ytr, Xte, n_gap):
    from sklearn.linear_model import LogisticRegression
    Xtr, Xte = zs(Xtr, Xte)
    cut = int(0.8 * len(ytr))
    best = (-1, Cs[0])
    if len(np.unique(ytr[cut + n_gap:])) > 1:
        for C in Cs:                                  # inner block: last 20 % of training data
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=300)
            m.fit(Xtr[:cut], ytr[:cut])
            best = max(best, (bal_acc(ytr[cut + n_gap:], m.predict(Xtr[cut + n_gap:]),
                                      np.unique(ytr)), C))
    m = LogisticRegression(C=best[1], class_weight="balanced", max_iter=500).fit(Xtr, ytr)
    return m.predict(Xte), m.predict_proba(Xte), m.classes_

def fit_ridge(Xtr, Ytr, Xte, n_gap):
    from sklearn.linear_model import Ridge
    Xtr, Xte = zs(Xtr, Xte)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-9
    Ytr = (Ytr - mu) / sd
    cut = int(0.8 * len(Ytr))
    err = []
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr[:cut], Ytr[:cut])
        err.append(((m.predict(Xtr[cut + n_gap:]) - Ytr[cut + n_gap:]) ** 2).mean(0))
    a_best = np.array(alphas)[np.argmin(np.array(err), 0)]
    P = np.column_stack([Ridge(alpha=a).fit(Xtr, Ytr[:, j]).predict(Xte)
                         for j, a in enumerate(a_best)])
    return P * sd + mu

def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)

def corr(Y, P):
    """Pearson r per column of the out-of-fold predictions with the truth, after centring both
    within each test block: scale-free, and free of the block offsets (a decoder predicting
    the training mean anti-correlates with each held-out block's own mean)."""
    edges = np.linspace(0, len(Y), n_folds + 1).astype(int)
    Yc, Pc = Y.astype(float).copy(), P.astype(float).copy()
    for a, b in zip(edges[:-1], edges[1:]):
        Yc[a:b] -= Yc[a:b].mean(0)
        Pc[a:b] -= Pc[a:b].mean(0)
    return (Yc * Pc).sum(0) / np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0) + 1e-12)

def cv_decode(X, y, Y, classes, n_gap):
    """Out-of-fold predictions: labels (pred, proba) and features from X."""
    n = len(y)
    pred, proba, Fh = np.zeros(n, int), np.zeros((n, len(classes))), np.zeros_like(Y)
    for tr, te in folds(n, n_gap):
        p, pr, cl = fit_logistic(X[tr], y[tr], X[te], n_gap)
        pred[te] = p
        proba[np.ix_(te, np.searchsorted(classes, cl))] = pr
        Fh[te] = fit_ridge(X[tr], Y[tr], X[te], n_gap)
    return pred, proba, Fh

def cv_labels(X, y, n_gap):
    n = len(y)
    pred = np.zeros(n, int)
    for tr, te in folds(n, n_gap):
        pred[te] = fit_logistic(X[tr], y[tr], X[te], n_gap)[0]
    return pred

def cv_features(X, Y, n_gap):
    n = len(Y)
    P = np.zeros_like(Y)
    for tr, te in folds(n, n_gap):
        P[te] = fit_ridge(X[tr], Y[tr], X[te], n_gap)
    return P

def recording(day, rec):
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(abs(hash((day, rec, seed))) % 2 ** 32)
    with h5py.File(dataset, "r") as h:
        g, r = h[day], h[day][rec]
        attrs = dict(day=day, rec=rec, condition=r.attrs["condition"], mouse=g.attrs["mouse"],
                     spn=g.attrs["spn"])
        F, lab, names = behaviour(r["behaviour"])
        C, sn = r["C"][:], r["sn"][:]
        ptr, fr = r["event_ptr"][:], r["event_frames"][:]
    n, T = C.shape
    E = np.zeros((n, T), np.float32)
    for i in range(n):
        E[i, fr[ptr[i]:ptr[i + 1]]] = 1
    reads = {"spikes": gaussian_filter1d(E, kernel_s * fs, axis=1), "trace": C / sn[:, None]}
    pad = int(max(abs(l) for l in lags_s) * fs)
    fin = np.isfinite(F).all(1)
    Xz = {k: (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9) for k, X in reads.items()}
    L = int(seg_s * fs)
    rows = []
    for seg in range(max(1, T // L)):                       # 15-min segments (Control: 1, Amph: 3)
        idx = np.arange(seg * L + pad, min(T, (seg + 1) * L) - pad, step)
        idx = idx[(lab[idx] >= 0) & np.all([fin[idx + int(round(l * fs))] for l in lags_s], 0)]
        rows += segment(idx, lab, F, Xz, names, T, n, rng, dict(attrs, segment=seg))
    print(day, rec, flush=True)
    return rows

def segment(idx, lab, F, Xz, names, T, n, rng, attrs):
    y = lab[idx]
    keep_cls = [c for c in np.unique(y) if (y == c).sum() * step / fs >= min_class_s]
    ok = np.isin(y, keep_cls)
    idx, y, Y = idx[ok], y[ok], F[idx[ok]]
    classes = np.array(keep_cls)
    n_gap = int(gap_s * fs / step)
    rows = []
    base = dict(attrs, n_cells=n, n_samples=len(idx), n_classes=len(classes),
                classes="|".join(names[c] for c in classes))
    # ceilings (behaviour only): labels from the true features, features from the true labels
    Fl = np.hstack([F[np.clip(idx + int(round(l * fs)), 0, T - 1)] for l in lags_s])
    ceil_lab = bal_acc(y, cv_labels(Fl, y, n_gap), classes)
    onehot = (y[:, None] == classes[None]).astype(float)
    P_ceil = cv_features(onehot, Y, n_gap)
    ceil_feat, ceil_feat_r2 = corr(Y, P_ceil), r2(Y, P_ceil)
    for key, X in Xz.items():
        for d in range(n_draw):
            sel = np.sort(rng.choice(n, min(n_sub, n), replace=False))
            Xl = lagged(X[sel], idx)
            pred, proba, Fh = cv_decode(Xl, y, Y, classes, n_gap)
            via_f = bal_acc(y, cv_labels(Fh, y, n_gap), classes)
            P_via = cv_features(proba, Y, n_gap)
            s = rng.integers(int(null_min_s * fs), T - int(null_min_s * fs))
            Xn = lagged(np.roll(X[sel], s, axis=1), idx)
            null_l = bal_acc(y, cv_labels(Xn, y, n_gap), classes)
            P_null = cv_features(Xn, Y, n_gap)
            b = dict(base, readout=key, draw=d)
            rows.append(dict(b, target="labels", direct=bal_acc(y, pred, classes), via=via_f,
                             ceiling=ceil_lab, null=null_l, chance=1 / len(classes)))
            for j, f in enumerate(features):
                rows.append(dict(b, target=f, direct=corr(Y, Fh)[j], via=corr(Y, P_via)[j],
                                 ceiling=ceil_feat[j], null=corr(Y, P_null)[j], chance=0.0,
                                 direct_r2=r2(Y, Fh)[j], via_r2=r2(Y, P_via)[j],
                                 ceiling_r2=ceil_feat_r2[j], null_r2=r2(Y, P_null)[j]))
    return rows

def main(n_jobs=4):
    from joblib import Parallel, delayed
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(dataset, "r") as h:
        todo = [(d, r) for d in h for r in ("rec0", "rec1")
                if h[d][r].attrs["condition"] in conditions
                and h[d][r]["behaviour"].attrs.get("available", False)]
    if os.environ.get("ONLY"):
        todo = todo[:int(os.environ["ONLY"])]
    res = Parallel(n_jobs=n_jobs)(delayed(recording)(d, r) for d, r in todo)
    df = pd.DataFrame([row for rows in res for row in rows])
    df.to_csv(f"{out_dir}/decoding.csv", index=False)
    m = df.groupby(["readout", "target", "day", "rec"]) \
        [["direct", "via", "ceiling", "null", "chance"]].mean().reset_index()
    pd.set_option("display.width", 200)
    print(m.groupby(["readout", "target"])[["direct", "via", "ceiling", "null", "chance"]]
          .median().round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
