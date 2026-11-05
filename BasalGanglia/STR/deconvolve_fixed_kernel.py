"""Re-deconvolve every cell with a fixed calcium kernel and re-detect events.

The pipeline's OASIS deconvolution uses each cell's own AR(1) coefficient from CNMF-E; with the
antipsychotics + amphetamine these fitted decay times are ~17-20 % longer than in Control (median
0.66 vs 0.56 s), while the event count drops more than the traces. Test: is the event-rate change
independent of the per-cell kernel fit? Here the raw extracted trace
(Yraw = C + YrA, from the recording's outputs file on the shared drive) is deconvolved with the
same OASIS call, noise level and event detection (summed over 3 frames >= 1.5 x noise, events
>= 0.5 s apart), but with the decay time fixed to tau_s for every cell (default 0.56 s: the
median CNMF-E AR(1) decay of the Control recordings, both cohorts). The events are added to the SPN data files as
event_frames_fixed / event_ptr_fixed (same CSR layout as event_frames / event_ptr), with the
kernel in the attrs.

  conda run -n parker python deconvolve_fixed_kernel.py [n_jobs]        (TAU_S=0.56)
"""
import os
import sys
import numpy as np
import h5py

import pipeline_raw_spikes as P

drive = "/mnt/kennedy_labdata/Parkerlab/results/raw_pipeline"
data_dir = "/home/rgast/data/parker_data/dataset"
tag, sel = "p1_c0.6_pnr5_nrg0.999", "snr2_sz_rim5_hyb0.3-0.2_ves3_off6_fedge1.05"
tau_s = float(os.environ.get("TAU_S", 0.56))
fs = 20.0
P.detect_method, P.sum_frames, P.spike_thr = "sum", 3, 1.5
drugs = ["Clozapine", "Haloperidol"]

def recording(name):
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
    z = np.load(f"{drive}/{name}/new_pipeline_outputs_{tag}_{sel}.npz")
    Y, sn = z["Yraw"], z["sn"]
    g = np.exp(-1 / (fs * tau_s))
    S = np.zeros_like(Y)
    for k in range(len(Y)):
        *_, sp, _ = constrained_foopsi(Y[k], g=np.array([g]), sn=sn[k], p=1, method_deconvolution="oasis")
        S[k] = sp
    ev = P.detect(S, sn, P.spike_thr)
    frames = [np.flatnonzero(e) for e in ev]
    return name, np.concatenate(frames).astype(np.int32), np.r_[0, np.cumsum([len(f) for f in frames])]

def main(n_jobs=4):
    from joblib import Parallel, delayed
    for drug in drugs:
        path = f"{data_dir}/{drug.lower()}_spn.h5"
        with h5py.File(path, "r") as h:
            todo = [f"{d}_{r}" for d in h for r in ("rec0", "rec1")
                    if "event_ptr_fixed" not in h[d][r] or "--force" in sys.argv]
        print(drug, len(todo), "recordings", flush=True)
        for i in range(0, len(todo), 4 * n_jobs):                 # write in chunks (resumable)
            res = Parallel(n_jobs=n_jobs)(delayed(recording)(n) for n in todo[i:i + 4 * n_jobs])
            with h5py.File(path, "a") as h:
                for name, fr, ptr in res:
                    r = h[name[:-5]][name[-4:]]
                    for k in ("event_frames_fixed", "event_ptr_fixed"):
                        if k in r:
                            del r[k]
                    r.create_dataset("event_frames_fixed", data=fr, compression="gzip")
                    r.create_dataset("event_ptr_fixed", data=ptr)
                    r.attrs["fixed_kernel_tau_s"] = tau_s
                    assert len(ptr) - 1 == r["C"].shape[0]
            print(drug, i + len(res), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4)
