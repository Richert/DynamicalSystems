"""Batch-extract per-neuron instantaneous firing rates with OASIS deconvolution.

Loops over all drugs / doses / conditions and, for every session, runs OASIS
sparse autoregressive deconvolution (Friedrich et al. 2017) on the calcium
traces to obtain a per-neuron, per-frame activity estimate `s`. `s` is a
non-negative amplitude at each frame, proportional to the number of spikes in
that frame's bin -- i.e. an (uncalibrated) instantaneous firing rate, not a
list of spike times. The stored trace is the instantaneous rate `s * fs`
(spikes/s, uncalibrated).

All sessions are written into a single HDF5 file. Layout:
  /                              (root; attrs: method, input_field, fs, ...)
    <drug>/<dose>/<session>      (group; attrs: drug, dose, condition,
      rates        dataset       mouse, session, fs, n_neurons, ...)
      [denoised]   dataset (optional)
Downstream analysis can load these rates instead of `dff x scaling`.

Run with the dedicated env:
  /home/rgast/conda/envs/spikes/bin/python extract_oasis_rates.py
"""

import os
import numpy as np
import h5py
from scipy.io import loadmat
from oasis.functions import deconvolve

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

path = "/home/rgast/data/parker_data/neural_data"
save_dir = "/home/rgast/data/parker_data"
out_path = f"{save_dir}/oasis_rates.h5"   # single HDF5 file collecting all sessions

# which drugs to process; None -> every drug subfolder found under `path`
drugs = None
# which doses to process; None -> every dose subfolder found per drug
doses = None

# input trace the deconvolution runs on:
#   "dff_traces_5hz"   -> dF/F
#   "cnmfe_traces_5hz" -> CNMF-E denoised traces
input_field = "cnmfe_traces_5hz"
fs = 5.0                       # sampling rate of the input traces (Hz)

# OASIS options
oasis_opts = {
    "ar_order": 1,             # 1 (AR1) or 2 (AR2) calcium model
    "penalty": 1,              # 1 -> L1 (graded), 0 -> L0 (sparse) sparsity penalty
}

# output options
save_denoised = False          # also store the denoised calcium trace `c`
overwrite = False              # re-extract sessions whose output already exists

# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def extract_rates(traces: np.ndarray) -> tuple:
    """Deconvolve every neuron. Returns (s, c) with the same shape as `traces`:
    s = per-frame inferred activity, c = denoised calcium trace."""
    g_init = (None,) if oasis_opts["ar_order"] == 1 else (None, None)
    s_all = np.zeros_like(traces, dtype=float)
    c_all = np.zeros_like(traces, dtype=float)
    n_bad = 0
    for i in range(traces.shape[0]):
        y = np.asarray(traces[i], dtype=float)
        try:
            c, s, _, _, _ = deconvolve(y, g=g_init, penalty=oasis_opts["penalty"])
        except Exception as e:
            print(f"    neuron {i}: deconvolution failed ({e}); left as zeros")
            n_bad += 1
            continue
        # OASIS can return non-finite values (e.g. degenerate/flat traces)
        # without raising; drop those neurons to zeros so they don't poison
        # downstream statistics.
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(c))):
            n_bad += 1
            continue
        s_all[i], c_all[i] = s, c
    if n_bad:
        print(f"    {n_bad}/{traces.shape[0]} neuron(s) had no valid "
              f"deconvolution and were set to zeros")
    return s_all, c_all


def main():
    drug_list = drugs if drugs is not None else sorted(
        d for d in os.listdir(path) if os.path.isdir(f"{path}/{d}"))

    n_done = 0
    with h5py.File(out_path, "a") as h5:
        # root-level metadata describing the whole extraction
        h5.attrs.update({
            "method": "OASIS deconvolution (Friedrich et al. 2017)",
            "quantity": "instantaneous firing rate = OASIS s * fs",
            "units": "spikes/s (uncalibrated, proportional to spike count/frame)",
            "input_field": input_field,
            "fs": fs,
            "ar_order": oasis_opts["ar_order"],
            "penalty": oasis_opts["penalty"],
        })

        for drug in drug_list:
            drug_dir = f"{path}/{drug}"
            dose_list = doses if doses is not None else sorted(
                d for d in os.listdir(drug_dir) if os.path.isdir(f"{drug_dir}/{d}"))

            for dose in dose_list:
                dose_dir = f"{drug_dir}/{dose}"
                if not os.path.isdir(dose_dir):
                    continue

                for session in sorted(os.listdir(dose_dir)):
                    session_dir = f"{dose_dir}/{session}"
                    if not os.path.isdir(session_dir):
                        continue

                    # derive mouse id + condition from the folder name (as in
                    # the original analysis pipeline)
                    _, mouse_id, *cond = session.split("_")
                    condition = "amph" if "amph" in cond else "veh"
                    # session folder names are unique within a dose (amph folders
                    # carry an "_amph" suffix); condition is kept in the attrs.
                    group_path = f"{drug}/{dose}/{session}"

                    if group_path in h5 and not overwrite:
                        print(f"[skip] {group_path} (exists)")
                        continue

                    mat_path = f"{session_dir}/{condition}_drug.mat"
                    if not os.path.isfile(mat_path):
                        print(f"[warn] {mat_path} not found; skipping")
                        continue

                    try:
                        inner = loadmat(mat_path, simplify_cells=True)[f"{condition}_drug"]
                        traces = np.asarray(inner[input_field], dtype=float)
                    except Exception as e:
                        print(f"[warn] failed to load {mat_path}: {e}")
                        continue

                    print(f"[run ] {group_path}: "
                          f"{traces.shape[0]} neurons x {traces.shape[1]} frames")
                    s_all, c_all = extract_rates(traces)
                    rates = s_all * fs  # instantaneous firing rate (spikes/s)

                    if group_path in h5:      # overwrite an existing entry
                        del h5[group_path]
                    grp = h5.create_group(group_path)
                    grp.create_dataset("rates", data=rates, compression="gzip")
                    if save_denoised:
                        grp.create_dataset("denoised", data=c_all, compression="gzip")
                    grp.attrs.update({
                        "drug": drug, "dose": dose, "condition": condition,
                        "mouse": mouse_id, "session": session,
                        "input_field": input_field, "fs": fs,
                        "ar_order": oasis_opts["ar_order"],
                        "penalty": oasis_opts["penalty"],
                        "n_neurons": rates.shape[0], "n_frames": rates.shape[1],
                    })
                    h5.flush()
                    n_done += 1

    print(f"\nDone. Extracted OASIS rates for {n_done} session(s) into {out_path}")


if __name__ == "__main__":
    main()
