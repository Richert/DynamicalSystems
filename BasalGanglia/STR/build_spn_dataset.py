"""One HDF5 file with everything the SPN analyses need (Clozapine, all doses).

Layout (one group per recording day, e.g. "20191205_m85_clo01_amph"):
  /<day>                attrs: mouse, spn (D1 / D2), drug, dose
  /<day>/rec0, /rec1    attrs: condition, minutes, n_cells, lag_frames (imaging vs. the old
                        5 Hz time base; nan if unknown), pose_session
      C                 denoised traces (cells x frames, float32, 20 Hz)
      S_norm            OASIS deconvolved activity / noise level (rate proxy, float32)
      sn                noise level per cell
      event_frames, event_ptr   spike events (summed detection, pipeline settings) per cell
                        as CSR: frames of cell i = event_frames[event_ptr[i]:event_ptr[i+1]]
      component, centroids_px, A_data / A_indices / A_indptr / A_shape, dims
                        CNMF-E component ids and footprints (pixels in F order) of the cells
      behaviour/        pose features, AR-HMM state and behaviour label per imaging frame
                        (imaging frame f <-> pose frame f - lag; -1 / nan where unavailable);
                        attrs: model (pose directory), labels (label names, index order)
  /<day>/tracking       rec0_cell, rec1_cell, similarity: cells tracked within the day
                        (row indices into rec0 / rec1)
Neural part: selection tag/sel of pipeline_raw_spikes.py, read from the recordings' output
files on the shared drive (members C, S, sn, component, centroids only). The behaviour part
is rebuilt separately (python build_spn_dataset.py behaviour), e.g. for a new AR-HMM.

  conda run -n parker python build_spn_dataset.py [neural | behaviour | all]
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import h5py

import pipeline_raw_spikes as P

drive = "/mnt/kennedy_labdata/Parkerlab/results/raw_pipeline"
local = "/home/rgast/data/parker_data/raw_pipeline"
tracking_dir = "/home/rgast/data/parker_data/cell_tracking"
pose_dir = os.environ.get("POSE_MODEL_DIR", "/home/rgast/data/parker_data/pose_arhmm_v5")
out = os.environ.get("DATASET", "/home/rgast/data/parker_data/dataset/clozapine_spn.h5")
tag, sel = "p1_c0.6_pnr5", "snr2_eta0.4_sz_rim5"
P.detect_method, P.sum_frames, P.spike_thr = "sum", 3, 2.0          # as in the pipeline batch
dose_of = {"clo01": "Vehicle", "clo05": "LowDose", "clo04": "HighDose"}
condition_of = {("Vehicle", 0): "Control", ("Vehicle", 1): "Amphetamine",
                ("LowDose", 0): "Clozapine low", ("LowDose", 1): "A + Clozapine low",
                ("HighDose", 0): "Clozapine high", ("HighDose", 1): "A + Clozapine high"}
behaviour_cols = ["speed", "turn", "head_angle", "bend", "body_len", "head_speed", "head_turn",
                  "ear_width", "hip_width", "wall_dist", "wall_facing", "interp_share"]
gz = dict(compression="gzip", compression_opts=4, shuffle=True)

def days():
    recs = sorted(os.path.basename(p) for p in glob.glob(f"{local}/*_rec?")
                  if os.path.isfile(f"{p}/kept_{tag}_{sel}.npy"))
    return sorted({r[:-5] for r in recs if r[:-5] + "_rec0" in recs and r[:-5] + "_rec1" in recs})

def mouse_spn():
    q = pd.read_csv(f"{drive}/qc_all_recordings.csv")
    return dict(zip(q.mouse, q.spn))

def pose_session(day, rec):
    base = day.replace("_m85_", "_m085_")
    return base if rec == 1 else base.replace("_amph", "")

def lag_of(rec_dir):
    f = f"{rec_dir}/summary_{tag}_thr2_sum3_{sel}.json"
    return float(json.load(open(f))["lag_frames"]) if os.path.isfile(f) else np.nan

def write_neural(h, day, spn):
    g = h.require_group(day)
    mouse = day.split("_")[1].replace("m85", "m085")
    dose = dose_of[day.split("_")[2]]
    g.attrs.update(mouse=mouse, spn=spn.get(mouse, ""), drug="Clozapine", dose=dose)
    for rec in (0, 1):
        name = f"{day}_rec{rec}"
        if f"rec{rec}" in g:
            del g[f"rec{rec}"]
        r = g.create_group(f"rec{rec}")
        z = np.load(f"{drive}/{name}/new_pipeline_outputs_{tag}_{sel}.npz")
        C, S, sn = z["C"].astype(np.float32), z["S"], z["sn"]
        ev = P.detect(S, sn, P.spike_thr)
        frames = [np.flatnonzero(e) for e in ev]
        r.create_dataset("C", data=C, chunks=(1, C.shape[1]), **gz)
        r.create_dataset("S_norm", data=(S / sn[:, None]).astype(np.float32),
                         chunks=(1, C.shape[1]), **gz)
        r.create_dataset("sn", data=sn)
        r.create_dataset("event_frames", data=np.concatenate(frames).astype(np.int32), **gz)
        r.create_dataset("event_ptr", data=np.r_[0, np.cumsum([len(f) for f in frames])])
        r.create_dataset("component", data=z["component"])
        r.create_dataset("centroids_px", data=z["centroids_px"])
        li = np.load(f"{local}/{name}/lens_inputs_{tag}.npz")
        import scipy.sparse as sp
        A = sp.csc_matrix((li["A_data"], li["A_indices"], li["A_indptr"]),
                          shape=tuple(li["A_shape"]))[:, z["component"]]
        for k, v in (("A_data", A.data.astype(np.float32)), ("A_indices", A.indices),
                     ("A_indptr", A.indptr), ("A_shape", np.array(A.shape)),
                     ("dims", np.asarray(li["dims"]))):
            r.create_dataset(k, data=v)
        r.attrs.update(condition=condition_of[(dose, rec)], minutes=C.shape[1] / P.fs / 60,
                       n_cells=len(C), fs=P.fs, lag_frames=lag_of(f"{local}/{name}"),
                       pose_session=pose_session(day, rec))
    if "tracking" in g:
        del g["tracking"]
    f = f"{tracking_dir}/{day}_pairs.csv"
    if os.path.isfile(f):
        t = pd.read_csv(f)
        tr = g.create_group("tracking")
        for k in ("rec0_cell", "rec1_cell", "similarity"):
            tr.create_dataset(k, data=t[k].values)

def write_behaviour(h, day):
    st = pd.read_csv(f"{pose_dir}/pose_states.csv", index_col="state")
    labels = list(dict.fromkeys(st.label)) if "label" in st else list(st.name)
    lmap = np.array([labels.index(l) for l in (st.label if "label" in st else st.name)])
    for rec in (0, 1):
        r = h[f"{day}/rec{rec}"]
        if "behaviour" in r:
            del r["behaviour"]
        b = r.create_group("behaviour")
        T = r["C"].shape[1]
        ps = r.attrs["pose_session"]
        lag = r.attrs["lag_frames"]
        lag = 0 if not np.isfinite(lag) else int(lag)
        idx = np.arange(T) - lag                          # pose frame of each imaging frame
        ff, fz = f"{pose_dir}/features_{ps}.csv.gz", f"{pose_dir}/states_{ps}.csv.gz"
        if not (os.path.isfile(ff) and os.path.isfile(fz)):
            b.attrs["available"] = False
            continue
        feat = pd.read_csv(ff)
        z = pd.read_csv(fz).state.values
        ok = (idx >= 0) & (idx < min(len(feat), len(z)))
        for c in behaviour_cols:
            if c in feat:
                v = np.full(T, np.nan, np.float32)
                v[ok] = feat[c].values[idx[ok]]
                b.create_dataset(c, data=v, **gz)
        s = np.full(T, -1, np.int16)
        s[ok] = z[idx[ok]]
        b.create_dataset("state", data=s, **gz)
        lab = np.full(T, -1, np.int16)
        lab[ok] = lmap[z[idx[ok]]]
        b.create_dataset("label", data=lab, **gz)
        b.attrs.update(available=True, model=pose_dir, labels=labels,
                       states=list(st.name), excluded=list(st.name[st.excluded]),
                       lag_used=lag)

def main(what="all"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    spn = mouse_spn()
    ds = days()
    with h5py.File(out, "a") as h:
        h.attrs.update(selection=f"{tag}_{sel}", detection="summed over 3 frames, 2 x noise",
                       fs=P.fs)
        for day in ds:
            if what in ("neural", "all"):
                write_neural(h, day, spn)
            if what in ("behaviour", "all"):
                write_behaviour(h, day)
            print(day, flush=True)
    print(f"saved {out} ({len(ds)} days)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
