"""Does the slow drift of the calcium traces affect spike inference? Server check.

For a sample of mice per drug (all four high-dose-protocol recordings each: Control, Amphetamine,
<drug> high, A + <drug> high), every cell's raw extracted trace (Yraw = C + YrA, outputs file of
pipeline_raw_spikes.py) is detrended by subtracting its slow baseline (rolling 10th percentile
over base_s, computed at 2 Hz and interpolated), then deconvolved with the same OASIS call as the
pipeline (the cell's CNMF-E AR(1) coefficient and noise level) and events are detected with the
pipeline rule (spike mass over 3 frames >= 1.5 x noise, >= 0.5 s apart). Per cell: event rate of
the pipeline (original) and after detrending.

Outputs (OUT_DIR): <recording>.csv per recording (cells), and with --summary the per-mouse
median rates and their change vs Control, original vs detrended (summary.csv, printed).

  python check_detrended_spikes.py [n_jobs] [n_mice_per_drug]      (run)
  python check_detrended_spikes.py --summary                        (after the run)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

root = os.environ.get("WORK_ROOT", "/media/storage/DATA/Parkerlab/results/raw_pipeline")
out_dir = os.environ.get("OUT_DIR", f"{root}/check_detrended_spikes")
tag, sel = "p1_c0.6_pnr5_nrg0.999", "snr2_sz_rim5_hyb0.3-0.2_ves3_off6_fedge1.05"
fs = 20.0
base_s = 30.0                       # window of the slow baseline
drugs = {"clo": "Clozapine", "hal": "Haloperidol"}
seed = 1

def condition(name):
    """(drug, condition) of a recording name <date>_<mouse>_<code><dose>_amph_rec<k>."""
    p = name.split("_")
    drug, dose, rec = drugs[p[2][:3]], p[2][3:], int(name[-1])
    if dose == "01":
        return drug, "Control" if rec == 0 else "Amphetamine"
    if dose == "04":
        return drug, f"{drug} high" if rec == 0 else f"A + {drug} high"
    return drug, None

def sample(n_mice):
    """Recordings of n_mice random mice per drug that have all four high-dose-protocol recordings."""
    recs = [os.path.basename(d) for d in glob.glob(f"{root}/*_???0?_amph_rec?")
            if os.path.isfile(f"{d}/new_pipeline_outputs_{tag}_{sel}.npz")]
    df = pd.DataFrame([dict(name=r, mouse=r.split("_")[1].replace("m85", "m085"),
                            drug=condition(r)[0], cond=condition(r)[1]) for r in recs])
    df = df[df.cond.notna()]
    rng = np.random.default_rng(seed)
    out = []
    for drug, g in df.groupby("drug"):
        full = [m for m, gm in g.groupby("mouse") if gm.cond.nunique() == 4]
        pick = rng.choice(full, min(n_mice, len(full)), replace=False)
        out += g[g.mouse.isin(pick)].drop_duplicates(["mouse", "cond"]).name.tolist()
    return sorted(out)

def recording(name):
    import h5py
    from scipy.ndimage import percentile_filter
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
    import pipeline_raw_spikes as P
    P.detect_method, P.sum_frames, P.spike_thr = "sum", 3, 1.5
    f = f"{out_dir}/{name}.csv"
    if os.path.isfile(f):
        return
    d = f"{root}/{name}"
    z = np.load(f"{d}/new_pipeline_outputs_{tag}_{sel}.npz")
    Y, S, sn, comp = z["Yraw"], z["S"], z["sn"], z["component"]
    with h5py.File(f"{d}/cnmfe_{tag}.hdf5", "r") as h:
        g = np.asarray(h["estimates/g"][()])
    g = g.reshape(len(g), -1)[comp, 0]
    T = Y.shape[1]
    step = int(fs / 2)
    t_lo = np.arange(0, T, step)
    S2 = np.zeros_like(Y)
    drift = np.zeros(len(Y))
    for k in range(len(Y)):
        b = percentile_filter(Y[k, ::step], 10, size=int(base_s * 2) + 1, mode="nearest")
        base = np.interp(np.arange(T), t_lo[:len(b)], b)
        drift[k] = (np.percentile(base, 95) - np.percentile(base, 5)) / sn[k]
        *_, sp, _ = constrained_foopsi(Y[k] - base, g=np.array([g[k]]), sn=sn[k], p=1,
                                       method_deconvolution="oasis")
        S2[k] = sp
    ev0 = P.detect(S, sn, P.spike_thr).sum(1)
    ev1 = P.detect(S2, sn, P.spike_thr).sum(1)
    drug, cond = condition(name)
    pd.DataFrame(dict(recording=name, mouse=name.split("_")[1].replace("m85", "m085"), drug=drug,
                      condition=cond, rate_orig=ev0 / (T / fs), rate_detr=ev1 / (T / fs),
                      drift_noise_units=drift)).to_csv(f, index=False)
    print("done", name, flush=True)

def summary():
    c = pd.concat([pd.read_csv(f) for f in glob.glob(f"{out_dir}/*_rec?.csv")], ignore_index=True)
    m = c.groupby(["drug", "mouse", "condition"])[["rate_orig", "rate_detr", "drift_noise_units"]].median()
    rows = []
    for (drug, mouse), g in m.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        if "Control" not in g.index:
            continue
        for cond in g.index:
            rows.append(dict(drug=drug, mouse=mouse, condition=cond,
                             rate_orig=g.loc[cond, "rate_orig"], rate_detr=g.loc[cond, "rate_detr"],
                             drift=g.loc[cond, "drift_noise_units"],
                             change_orig=100 * (g.loc[cond, "rate_orig"] / g.loc["Control", "rate_orig"] - 1),
                             change_detr=100 * (g.loc[cond, "rate_detr"] / g.loc["Control", "rate_detr"] - 1)))
    s = pd.DataFrame(rows)
    s.to_csv(f"{out_dir}/summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{c.recording.nunique()} recordings, {len(c)} cells, {s.mouse.nunique()} mice")
    print(s.groupby(["drug", "condition"])[["rate_orig", "rate_detr", "drift", "change_orig", "change_detr"]]
          .median().round(3).to_string())


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    if "--summary" in sys.argv:
        summary()
    else:
        from joblib import Parallel, delayed
        n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
        n_mice = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        todo = sample(n_mice)
        print(f"{len(todo)} recordings", flush=True)
        Parallel(n_jobs=n_jobs)(delayed(recording)(n) for n in todo)
        summary()
