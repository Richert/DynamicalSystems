"""Are drug effects on SPN activity explained by the change in behaviour? Behaviour
re-weighting per mouse, spikes and denoised traces (from the SPN data file).

Per recording and behaviour category l (the labels of label_clean, plus "unlabelled" frames:
artefacts / bouts < 0.2 s, so that every frame is covered): occupancy occ(l) (share of
frames) and population activity a(l) (mean over cells of the event rate, or of the mean
denoised trace C / noise, over the frames of l). The session activity is exactly
A = sum_l occ(l) a(l). For a drug session D of a mouse with Control session C:
  expected  E = sum_l occ_D(l) a_C(l)       (Control activity per behaviour, drug behaviour)
  total     log10(A_D / A_C) = log10(E / A_C)  [behaviour]  +  log10(A_D / E)  [beyond behaviour]
a_C(l) needs >= min_s of l in the Control session; otherwise it is imputed as the mouse's
Control walk activity x the pooled Control ratio a(l) / a(walk) of that SPN type (the share of
drug-session time with imputed values is reported; beyond_measured: the beyond-behaviour term
over the categories with measured Control activity only). Amph + Clozapine sessions are compared to
the same mouse's Vehicle-day Control (different day, different cells).

Outputs (OUT_DIR): expected_by_behaviour.csv (per mouse x drug x readout), stats, figure.

  conda run -n parker python analysis_spn_behavior_expected.py
"""
import os
import numpy as np
import pandas as pd
import h5py
from scipy.stats import wilcoxon

dataset = os.environ.get("DATASET", "/home/rgast/data/parker_data/dataset/clozapine_spn.h5")
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/spn_behavior_expected")
CTRL, AMPH, ACLZ = "Control", "Amphetamine", "A + Clozapine high"
drugs = [AMPH, ACLZ]
fs = 20.0
min_s = 10.0

def recording_table(r):
    """Per category: seconds and population activity (spikes: Hz, trace: noise units)."""
    beh = r["behaviour"]
    names = list(beh.attrs["label_names"])
    lab = beh["label_clean"][:]
    C, sn = r["C"][:], r["sn"][:]
    ptr, fr = r["event_ptr"][:], r["event_frames"][:]
    n, T = C.shape
    lab = lab[:T]
    cats = {nm: lab == k for k, nm in enumerate(names)}
    cats["unlabelled"] = lab < 0
    ev_lab = [lab[fr[ptr[i]:ptr[i + 1]]] for i in range(n)]
    rows = []
    for nm, m in cats.items():
        secs = m.sum() / fs
        if secs == 0:
            continue
        k = names.index(nm) if nm in names else -1
        if nm == "unlabelled":
            cnt = np.array([(e < 0).sum() for e in ev_lab])
        else:
            cnt = np.array([(e == k).sum() for e in ev_lab])
        rows.append(dict(category=nm, seconds=secs, spikes=np.mean(cnt / secs),
                         trace=np.mean((C[:, m] / sn[:, None]).mean(1))))
    d = pd.DataFrame(rows)
    d["occ"] = d.seconds / d.seconds.sum()
    return d

def main():
    os.makedirs(out_dir, exist_ok=True)
    tabs = []
    with h5py.File(dataset, "r") as h:
        for day in h:
            for rec in ("rec0", "rec1"):
                r = h[day][rec]
                cond = r.attrs["condition"]
                if cond not in [CTRL] + drugs or not r["behaviour"].attrs.get("available", False):
                    continue
                t = recording_table(r)
                t["condition"], t["mouse"], t["spn"], t["day"] = cond, h[day].attrs["mouse"], h[day].attrs["spn"], day
                tabs.append(t)
            print(day, flush=True)
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(f"{out_dir}/activity_by_behaviour.csv", index=False)
    rows = []
    for readout in ("spikes", "trace"):
        # pooled Control ratio to walk per SPN type (seconds-weighted over mice)
        ctl = tab[tab.condition == CTRL]
        pooled = {}
        for spn, g in ctl.groupby("spn"):
            w = g.groupby("category").apply(lambda x: np.average(x[readout], weights=x.seconds))
            pooled[spn] = w / w["walk"]
        for (mouse, spn), g in tab.groupby(["mouse", "spn"]):
            c = g[g.condition == CTRL].set_index("category")
            if not len(c):
                continue
            A_C = (c.occ * c[readout]).sum()
            a_C = {}
            for cat in set(tab.category):
                if cat in c.index and c.seconds[cat] >= min_s:
                    a_C[cat] = c[readout][cat]
                else:
                    a_C[cat] = np.nan
            for drug in drugs:
                d = g[g.condition == drug].set_index("category")
                if not len(d):
                    continue
                A_D = (d.occ * d[readout]).sum()
                imputed = [cat for cat in d.index if not np.isfinite(a_C[cat])]
                aC = pd.Series({cat: a_C[cat] if np.isfinite(a_C[cat])
                                else a_C["walk"] * pooled[spn].get(cat, np.nan) for cat in d.index})
                E = (d.occ * aC).sum()
                ok = [cat for cat in d.index if cat not in imputed]          # measured Control only
                beyond_meas = np.log10((d.occ[ok] * d[readout][ok]).sum() / (d.occ[ok] * aC[ok]).sum())
                rows.append(dict(readout=readout, mouse=mouse, spn=spn, drug=drug, A_C=A_C, A_D=A_D,
                                 E=E, total=np.log10(A_D / A_C), behaviour=np.log10(E / A_C),
                                 beyond=np.log10(A_D / E), beyond_measured=beyond_meas,
                                 imputed_share=d.occ[imputed].sum() if imputed else 0.0))
    res = pd.DataFrame(rows)
    res.to_csv(f"{out_dir}/expected_by_behaviour.csv", index=False)
    st = []
    for (ro, spn, drug), g in res.groupby(["readout", "spn", "drug"]):
        for comp in ("total", "behaviour", "beyond", "beyond_measured"):
            v = g[comp].values
            st.append(dict(readout=ro, spn=spn, drug=drug, component=comp, n=len(v), mean=v.mean(),
                           sem=v.std(ddof=1) / np.sqrt(len(v)), p=wilcoxon(v).pvalue))
    st = pd.DataFrame(st)
    st.to_csv(f"{out_dir}/expected_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(st.round(4).to_string(index=False))
    print(res.groupby(["spn", "drug"]).imputed_share.describe().round(3).to_string())


if __name__ == "__main__":
    main()
