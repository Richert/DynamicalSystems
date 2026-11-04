"""Statistics of SPN activity per behaviour label: mixed models over all cells / recordings and
the within-cell Control vs Amphetamine comparison on tracked cells (from the SPN data file and
the readouts of analysis_spn_behavior_readouts.py).

Per cell, recording and label (plus "whole session"; recordings with >= T in the label, T the
time budget of the readouts):
  log rate   log10(events / s in the label), floored at 0.5 events / T  (spike-based)
  log trace  log10(mean denoised trace C / noise in the label), floored at trace_floor
                                                                          (trace-based)
Mixed models (statsmodels MixedLM, REML), per label and D1 / D2 separately:
  cell level       log rate, log trace ~ condition + (1 | mouse) + (1 | recording)
  recording level  SD of log rate, within-bout correlation, participation ratio
                   (readouts_<A_DRUG>.csv) ~ condition + (1 | mouse)
  contrasts Amph - Control, A + Cloz - Control, A + Cloz - Amph; Benjamini-Hochberg FDR over
  labels x D1 / D2 x contrasts within each readout.
Tracked cells (Vehicle days: Control = rec0, Amphetamine = rec1, cells tracked within the day):
  per cell delta = Amph - Control of log rate and log trace; mixed model delta ~ 1 + (1 | mouse);
  share of cells up / down.

Outputs (OUT_DIR): stats_mixed_<A_DRUG>.csv, stats_tracked.csv, cell_values.csv.gz, figures.

  conda run -n parker python analysis_spn_behavior_stats.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import h5py
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

dataset = os.environ.get("DATASET", "/home/rgast/data/parker_data/dataset/clozapine_spn.h5")
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/spn_behavior_states")
a_drug = os.environ.get("A_DRUG", "high")
fs = 20.0
CTRL, AMPH, ACLZ = "Control", "Amphetamine", f"A + Clozapine {a_drug}"
conditions = [CTRL, AMPH, ACLZ]
trace_floor = 0.01                         # noise units
min_bouts = 10
contrasts = [("Amph - Control", AMPH, CTRL), ("A+Cloz - Control", ACLZ, CTRL),
             ("A+Cloz - Amph", ACLZ, AMPH)]
order = ["whole session", "rest", "walk", "scanning walk", "sharp turn (CW)", "turning run (CCW)",
         "straight run", "turning run (CW)", "wall run"]

def bouts_of(lab):
    st = np.flatnonzero(np.diff(np.r_[-2, lab]) != 0)
    en = np.r_[st[1:], len(lab)]
    return st, en, lab[st]

def cell_values(T_lab):
    """Per cell, recording and label: log rate and log trace (all conditions incl. both
    recordings of the Vehicle days)."""
    out = []
    with h5py.File(dataset, "r") as h:
        for day in h:
            g = h[day]
            for rec in ("rec0", "rec1"):
                r = g[rec]
                cond = r.attrs["condition"]
                if cond not in conditions or not r["behaviour"].attrs.get("available", False):
                    continue
                lab = r["behaviour/label_clean"][:]
                names = list(r["behaviour"].attrs["label_names"])
                C, sn = r["C"][:], r["sn"][:]
                ptr, fr = r["event_ptr"][:], r["event_frames"][:]
                n, T = C.shape
                st, en, bl = bouts_of(lab)
                masks = {"whole session": (np.ones(T, bool), 10 ** 9)}
                for k, name in enumerate(names):
                    m = np.zeros(T, bool)
                    for a, b in zip(st[bl == k], en[bl == k]):
                        m[a:b] = True
                    masks[name] = (m, int((bl == k).sum()))
                for name, (m, nb) in masks.items():
                    if name not in T_lab or m.sum() / fs < T_lab[name] or nb < min_bouts:
                        continue
                    secs = m.sum() / fs
                    cnt = np.array([m[fr[ptr[i]:ptr[i + 1]]].sum() for i in range(n)])
                    tr = (C[:, m] / sn[:, None]).mean(1)
                    out.append(pd.DataFrame(dict(
                        day=day, rec=rec, recording=f"{day}_{rec}", condition=cond,
                        mouse=g.attrs["mouse"], spn=g.attrs["spn"], label=name, cell=np.arange(n),
                        log_rate=np.log10(np.maximum(cnt / secs, 0.5 / T_lab[name])),
                        log_trace=np.log10(np.maximum(tr, trace_floor)))))
            print(day, flush=True)
    return pd.concat(out, ignore_index=True)

def fit(d, y, cell_level):
    """Mixed model y ~ condition; returns {contrast: (estimate, se, p)}."""
    d = d[d[y].notna()].copy()
    n = d.groupby("condition").recording.nunique() if cell_level else d.condition.value_counts()
    present = [c for c in conditions if n.get(c, 0) >= 2]    # >= 2 recordings per condition
    d = d[d.condition.isin(present)]
    have = set(present)
    if len(have) < 2 or d.mouse.nunique() < 3:
        return {}
    ref = present[0]
    d["cond"] = pd.Categorical(d.condition, categories=present)
    kw = dict(groups="mouse", re_formula="1")
    if cell_level:
        kw["vc_formula"] = {"recording": "0 + C(recording)"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm(f"{y} ~ C(cond, Treatment('{ref}'))", d, **kw).fit(reml=True)
    except (np.linalg.LinAlgError, ValueError):
        return {}
    coef = {ref: np.zeros(len(m.fe_params))}
    for c in present[1:]:
        v = np.zeros(len(m.fe_params))
        key = f"C(cond, Treatment('{ref}'))[T.{c}]"
        if key in m.fe_params.index:
            v[list(m.fe_params.index).index(key)] = 1
            coef[c] = v
    res = {}
    for name, a, b in contrasts:
        if a in coef and b in coef and a in have and b in have:
            t = m.t_test((coef[a] - coef[b])[None])
            res[name] = (float(t.effect[0]), float(t.sd[0, 0]), float(t.pvalue))
    return res

def mixed_tables(cv, rd):
    rows = []
    specs = [("log_rate", cv, True), ("log_trace", cv, True),
             ("sd_log_rate", rd, False), ("within_bout_corr", rd, False),
             ("participation_ratio", rd, False)]
    for y, d, cell_level in specs:
        for (label, spn), g in d.groupby(["label", "spn"]):
            n = g.groupby("condition").mouse.nunique()
            for name, (est, se, p) in fit(g, y, cell_level).items():
                a, b = [c for n_, c, _ in contrasts if n_ == name][0], \
                       [c for n_, _, c in contrasts if n_ == name][0]
                rows.append(dict(readout=y, label=label, spn=spn, contrast=name, estimate=est,
                                 se=se, p=p, n_mice_a=n.get(a, 0), n_mice_b=n.get(b, 0)))
    st = pd.DataFrame(rows)
    st["p_fdr"] = np.nan
    for y, g in st.groupby("readout"):
        ok = g.p.notna()
        st.loc[g.index[ok], "p_fdr"] = multipletests(g.p[ok], method="fdr_bh")[1]
    return st

def tracked(cv):
    """Within-cell Amph - Control on the Vehicle days."""
    rows = []
    with h5py.File(dataset, "r") as h:
        pairs = {d: (h[d]["tracking/rec0_cell"][:], h[d]["tracking/rec1_cell"][:])
                 for d in h if "tracking" in h[d] and h[d].attrs["dose"] == "Vehicle"}
    cells = []
    for day, (i0, i1) in pairs.items():
        for label, g in cv[cv.day == day].groupby("label"):
            a, b = g[g.rec == "rec0"].set_index("cell"), g[g.rec == "rec1"].set_index("cell")
            if not len(a) or not len(b):
                continue
            cells.append(pd.DataFrame(dict(day=day, mouse=a.mouse.iloc[0], spn=a.spn.iloc[0],
                                           label=label, ctrl_rate=a.log_rate.loc[i0].values,
                                           amph_rate=b.log_rate.loc[i1].values,
                                           ctrl_trace=a.log_trace.loc[i0].values,
                                           amph_trace=b.log_trace.loc[i1].values)))
    tc = pd.concat(cells, ignore_index=True)
    tc["d_rate"] = tc.amph_rate - tc.ctrl_rate
    tc["d_trace"] = tc.amph_trace - tc.ctrl_trace
    for (label, spn), g in tc.groupby(["label", "spn"]):
        for y in ("d_rate", "d_trace"):
            r = dict(readout=y, label=label, spn=spn, n_cells=len(g), n_mice=g.mouse.nunique(),
                     median=g[y].median(), share_up=np.mean(g[y] > 0), share_down=np.mean(g[y] < 0))
            if g.mouse.nunique() >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = smf.mixedlm(f"{y} ~ 1", g, groups="mouse").fit(reml=True)
                r.update(estimate=m.fe_params.iloc[0], se=m.bse.iloc[0], p=m.pvalues.iloc[0])
            rows.append(r)
    ts = pd.DataFrame(rows)
    ok = ts.p.notna()
    ts.loc[ok, "p_fdr"] = multipletests(ts.p[ok], method="fdr_bh")[1]
    return tc, ts


def main():
    rd = pd.read_csv(f"{out_dir}/readouts_{a_drug}.csv")
    T_lab = rd.groupby("label")["T"].first().to_dict()
    f = f"{out_dir}/cell_values_{a_drug}.csv.gz"
    if os.path.isfile(f):
        cv = pd.read_csv(f)
    else:
        cv = cell_values(T_lab)
        cv.to_csv(f, index=False)
    st = mixed_tables(cv, rd)
    st.to_csv(f"{out_dir}/stats_mixed_{a_drug}.csv", index=False)
    tc, ts = tracked(cv)
    tc.to_csv(f"{out_dir}/tracked_cells.csv.gz", index=False)
    ts.to_csv(f"{out_dir}/stats_tracked.csv", index=False)
    pd.set_option("display.width", 220)
    show = st.copy()
    show["label"] = pd.Categorical(show.label, order)
    print(show.sort_values(["readout", "label", "spn"]).round(4).to_string(index=False))
    ts["label"] = pd.Categorical(ts.label, order)
    print(ts.sort_values(["readout", "label", "spn"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
