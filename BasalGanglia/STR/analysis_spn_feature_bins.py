"""SPN rate statistics conditioned on speed and turning bins, per drug condition (both cohorts; SPN
data files of build_spn_dataset.py).

Bins (frames):
  speed     rest < 0.5 cm/s (as Yun et al. 2023), slow 0.5-3 cm/s, fast >= 3 cm/s
  turning   moving frames only (speed >= 0.5 cm/s): left > +turn_thr, straight |turn| <= turn_thr,
            right < -turn_thr (deg/s; sign as in the top-view video: + = counter-clockwise)
Runs of a bin shorter than min_run_s are ignored. Equal-time / equal-neuron sampling as in
analysis_spn_behavior_readouts.py: per bin a time budget T (smallest median time over the
conditions, clipped to [t_min, t_max]; recordings with less time left out); n_rep draws of n_sub
cells and of runs up to T; per draw the mean and SD over cells of log10(events / T) (zero counts
floored at 0.5 events), averaged over draws.
Mixed models per bin and SPN type (recording level: readout ~ condition + (1 | mouse)), contrasts
vs Control and A + drug vs Amphetamine, BH-FDR within readout x cohort.

Outputs (OUT_DIR): readouts_<drug>.csv, stats.csv.

  conda run -n parker python analysis_spn_feature_bins.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import h5py

data_dir = "/home/rgast/data/parker_data/dataset"
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/spn_feature_bins")
drugs = ["Clozapine", "Haloperidol"]
fs = 20.0
rest_cm_s, fast_cm_s, turn_thr = 0.5, 3.0, 30.0
min_run_s = 0.25
t_min, t_max = 20.0, 120.0
n_sub, n_rep = 50, 50
seed = 1
BINS = [("speed", "rest"), ("speed", "slow"), ("speed", "fast"),
        ("turning", "left"), ("turning", "straight"), ("turning", "right")]

def conditions(drug):
    return ["Control", "Amphetamine", f"{drug} high", f"A + {drug} high"]

def masks(b):
    sp, tu = b["speed"][:], b["turn"][:]
    ok = np.isfinite(sp) & np.isfinite(tu)
    mov = ok & (sp >= rest_cm_s)
    return {("speed", "rest"): ok & (sp < rest_cm_s), ("speed", "slow"): mov & (sp < fast_cm_s),
            ("speed", "fast"): ok & (sp >= fast_cm_s),
            ("turning", "left"): mov & (tu > turn_thr), ("turning", "straight"): mov & (np.abs(tu) <= turn_thr),
            ("turning", "right"): mov & (tu < -turn_thr)}

def runs(m):
    d = np.diff(np.r_[0, m.astype(int), 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (en - st) >= int(min_run_s * fs)
    return st[keep], en[keep]

def budgets(drug):
    rows = []
    with h5py.File(f"{data_dir}/{drug.lower()}_spn.h5", "r") as h:
        for d in h:
            for rec in ("rec0", "rec1"):
                r = h[d][rec]
                if r.attrs["condition"] not in conditions(drug) or not r["behaviour"].attrs.get("available", False):
                    continue
                for k, m in masks(r["behaviour"]).items():
                    st, en = runs(m)
                    rows.append(dict(condition=r.attrs["condition"], bin=k, secs=(en - st).sum() / fs))
    d = pd.DataFrame(rows).groupby(["bin", "condition"]).secs.median().groupby("bin").min()
    return d.clip(t_min, t_max).to_dict()

def draw(st, en, budget, rng):
    out, tot = [], 0
    for i in rng.permutation(len(st)):
        a, b = st[i], min(en[i], st[i] + budget - tot)
        out.append((a, b))
        tot += b - a
        if tot >= budget:
            return out
    return None

def main():
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    allrows = []
    for drug in drugs:
        T_bin = budgets(drug)
        rows = []
        with h5py.File(f"{data_dir}/{drug.lower()}_spn.h5", "r") as h:
            for day in h:
                for rec in ("rec0", "rec1"):
                    r = h[day][rec]
                    cond = r.attrs["condition"]
                    if cond not in conditions(drug) or not r["behaviour"].attrs.get("available", False):
                        continue
                    ptr, fr = r["event_ptr"][:], r["event_frames"][:]
                    n, T = len(ptr) - 1, r["C"].shape[1]
                    cs = np.zeros((n, T + 1), np.float32)
                    for i in range(n):
                        cs[i, fr[ptr[i]:ptr[i + 1]] + 1] += 1
                    cs = np.cumsum(cs, 1)
                    for k, m in masks(r["behaviour"]).items():
                        budget = int(round(T_bin[k] * fs))
                        st, en = runs(m[:T])
                        if (en - st).sum() < budget:
                            continue
                        ml, sl = [], []
                        for _ in range(n_rep):
                            sel = rng.choice(n, min(n_sub, n), replace=False)
                            cnt = sum(cs[sel, b] - cs[sel, a] for a, b in draw(st, en, budget, rng))
                            lr = np.log10(np.maximum(cnt, 0.5) / T_bin[k])
                            ml.append(lr.mean())
                            sl.append(lr.std())
                        rows.append(dict(drug=drug, day=day, rec=rec, condition=cond, mouse=h[day].attrs["mouse"],
                                         spn=h[day].attrs["spn"], feature=k[0], bin=k[1], T=T_bin[k],
                                         secs=(en - st).sum() / fs, mean_log_rate=np.mean(ml),
                                         sd_log_rate=np.mean(sl)))
                print(drug, day, flush=True)
        rd = pd.DataFrame(rows)
        rd.to_csv(f"{out_dir}/readouts_{drug.lower()}.csv", index=False)
        allrows.append(rd)
    st = stats(pd.concat(allrows, ignore_index=True))
    st.to_csv(f"{out_dir}/stats.csv", index=False)
    pd.set_option("display.width", 220)
    rd = pd.concat(allrows, ignore_index=True)
    print(rd.groupby(["drug", "feature", "bin", "spn", "condition"])[["mean_log_rate", "sd_log_rate"]]
          .mean().round(3).unstack().to_string())
    print(st[st.p_fdr < 0.05].round(4).to_string(index=False))

def stats(rd):
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    out = []
    for (drug, feat, b, spn), d in rd.groupby(["drug", "feature", "bin", "spn"]):
        conds = conditions(drug)
        present = [c for c in conds if (d.condition == c).sum() >= 3]
        if "Control" not in present or d.mouse.nunique() < 3:
            continue
        d = d[d.condition.isin(present)].copy()
        d["cond"] = pd.Categorical(d.condition, present)
        for y in ("mean_log_rate", "sd_log_rate"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fit = smf.mixedlm(f"{y} ~ C(cond, Treatment('Control'))", d, groups="mouse").fit(reml=True)
                except Exception:
                    continue
            names = list(fit.fe_params.index)
            def v(c):
                key = f"C(cond, Treatment('Control'))[T.{c}]"
                e = np.zeros(len(names))
                if key in names:
                    e[names.index(key)] = 1
                return e if key in names or c == "Control" else None
            for cname, a_, b_ in (("Amph - Control", conds[1], "Control"), ("drug - Control", conds[2], "Control"),
                                  ("A+drug - Control", conds[3], "Control"), ("A+drug - Amph", conds[3], conds[1])):
                va, vb = v(a_), v(b_)
                if va is None or vb is None:
                    continue
                t = fit.t_test((va - vb)[None])
                out.append(dict(drug=drug, feature=feat, bin=b, spn=spn, readout=y, contrast=cname,
                                estimate=float(t.effect[0]), p=float(t.pvalue)))
    st = pd.DataFrame(out)
    st["p_fdr"] = np.nan
    for _, g in st.groupby(["drug", "readout"]):
        st.loc[g.index, "p_fdr"] = multipletests(g.p, method="fdr_bh")[1]
    return st


if __name__ == "__main__":
    main()
