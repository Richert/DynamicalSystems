"""One HDF5 file per drug with everything the SPN analyses need (DRUG, default Clozapine; all doses).

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
                        (imaging frame f <-> video frame video_offset + f, i.e. feature row
                        video_offset + f - pose offset_frames; -1 / nan where unavailable);
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

def _display_names(model_dir):
    """Label display names of the pose model version (analysis_pose_arhmm.model_cfg)."""
    import importlib, analysis_pose_arhmm as A
    cfg = A.model_cfg.get(os.path.basename(os.path.normpath(model_dir)), {})
    return {k: v[0] for k, v in cfg.get("display", {}).items()}

drive = "/mnt/kennedy_labdata/Parkerlab/results/raw_pipeline"
local = "/home/rgast/data/parker_data/raw_pipeline"
tracking_dir = "/home/rgast/data/parker_data/cell_tracking"
drug = os.environ.get("DRUG", "Clozapine")
code = drug[:3].lower()                  # recording names: <date>_<mouse>_<code><dose>_amph
_sub = "" if drug == "Clozapine" else f"{drug.lower()}/"
pose_dir = os.environ.get("POSE_MODEL_DIR", f"/home/rgast/data/parker_data/{_sub}pose_arhmm_v7")
out = os.environ.get("DATASET", f"/home/rgast/data/parker_data/dataset/{drug.lower()}_spn.h5")
if drug != "Clozapine":
    tracking_dir += f"_{drug.lower()}"
tag, sel = "p1_c0.6_pnr5_nrg0.999", "snr2_sz_rim5_hyb0.3-0.2_ves3_off6_fedge1.05"
P.detect_method, P.sum_frames, P.spike_thr = "sum", 3, 1.5          # as in the final full run
dose_of = {"clo01": "Vehicle", "clo05": "LowDose", "clo04": "HighDose",
           "hal01": "Vehicle", "hal03": "LowDose", "hal04": "HighDose"}
condition_of = {("Vehicle", 0): "Control", ("Vehicle", 1): "Amphetamine",
                ("LowDose", 0): f"{drug} low", ("LowDose", 1): f"A + {drug} low",
                ("HighDose", 0): f"{drug} high", ("HighDose", 1): f"A + {drug} high"}
behaviour_cols = ["speed", "turn", "head_angle", "bend", "body_len", "head_speed", "head_turn",
                  "ear_width", "hip_width", "wall_dist", "wall_facing", "wall_ahead", "interp_share"]
# Imaging frame 0 = video frame 12007 in every session (TTL camera started 600.35 s before
# imaging): from the motion-correction shifts vs keypoint movement (frame-precise in the
# sessions where the shifts track movement; 12005-12008 there). The per-session offsets from
# the 5 Hz speed match (pose_imaging_offsets.csv) scatter by ~+-0.5 s around it (up to 3.4 s)
# and only give the start of the feature files.
video_offset = 12007
offsets_csv = "/home/rgast/data/parker_data/pose_alignment/pose_imaging_offsets" + \
    ("" if drug == "Clozapine" else f"_{drug.lower()}") + ".csv"
gz = dict(compression="gzip", compression_opts=4, shuffle=True)

def days():
    recs = sorted(os.path.basename(p) for p in glob.glob(f"{local}/*_{code}0?_amph_rec?")
                  if os.path.isfile(f"{p}/kept_{tag}_{sel}.npy"))
    return sorted({r[:-5] for r in recs if r[:-5] + "_rec0" in recs and r[:-5] + "_rec1" in recs})

def mouse_spn():
    q = pd.read_csv(f"{drive}/qc_all_recordings.csv")
    return dict(zip(q.mouse, q.spn))

def pose_session(day, rec):
    base = day.replace("_m85_", "_m085_")
    return base if rec == 1 else base.replace("_amph", "")

def lag_of(rec_dir):
    f = f"{rec_dir}/summary_{tag}_thr1.5_sum3_{sel}.json"
    return float(json.load(open(f))["lag_frames"]) if os.path.isfile(f) else np.nan

def write_neural(h, day, spn):
    g = h.require_group(day)
    mouse = day.split("_")[1].replace("m85", "m085")
    dose = dose_of[day.split("_")[2]]
    g.attrs.update(mouse=mouse, spn=spn.get(mouse, ""), drug=drug, dose=dose)
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
    offs = pd.read_csv(offsets_csv)
    offs = dict(zip(offs.session.str.split("/").str[-1], offs.offset_frames))
    excl_idx = sorted({lmap[k] for k in st.index if st.excluded[k]})
    for rec in (0, 1):
        r = h[f"{day}/rec{rec}"]
        if "behaviour" in r:
            del r["behaviour"]
        b = r.create_group("behaviour")
        T = r["C"].shape[1]
        ps = r.attrs["pose_session"]
        ff, fz = f"{pose_dir}/features_{ps}.csv.gz", f"{pose_dir}/states_{ps}.csv.gz"
        if not (os.path.isfile(ff) and os.path.isfile(fz)) or ps not in offs:
            b.attrs["available"] = False
            continue
        shift = video_offset - int(offs[ps])               # feature row of imaging frame 0
        idx = np.arange(T) + shift
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
        lab[np.isin(lab, excl_idx)] = -1                  # excluded labels (artefacts): -1
        b.create_dataset("label", data=lab, **gz)
        zc = pd.read_csv(fz).state_clean.values if "state_clean" in pd.read_csv(fz, nrows=1) else z
        labc = np.full(T, -1, np.int16)                   # bouts < min_bout_s unlabelled too
        okc = ok & (zc[np.clip(idx, 0, len(zc) - 1)] >= 0)
        labc[okc] = lmap[zc[idx[okc]]]
        labc[np.isin(labc, excl_idx)] = -1
        b.create_dataset("label_clean", data=labc, **gz)
        b.attrs.update(available=True, model=pose_dir, labels=labels,
                       states=list(st.name), excluded=list(st.name[st.excluded]),
                       label_names=[_display_names(pose_dir).get(l, l) for l in labels],
                       video_offset=video_offset, row_shift=shift)

def main(what="all"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    spn = mouse_spn()
    ds = days()
    with h5py.File(out, "a") as h:
        h.attrs.update(selection=f"{tag}_{sel}", detection="summed over 3 frames, 1.5 x noise",
                       fs=P.fs)
        for day in ds:
            done = day in h and all(f"rec{r}/behaviour" in h[day] for r in (0, 1))
            if what == "all" and done and "--force" not in sys.argv:
                continue                                  # resume an interrupted build
            if what in ("neural", "all"):
                write_neural(h, day, spn)
            if what in ("behaviour", "all"):
                write_behaviour(h, day)
            print(day, flush=True)
    print(f"saved {out} ({len(ds)} days)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
