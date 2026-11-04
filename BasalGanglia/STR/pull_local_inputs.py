"""Copy the small per-recording inputs of the local analyses (cell tracking, data file, slide
figures) from the shared drive: from cnmfe_<tag>.hdf5 the footprints A (sparse), baseline b0,
noise image sn and component SNR; from new_pipeline_outputs_<tag>_<sel>.npz the kept
components and their centroids. Writes lens_inputs_<tag>.npz and kept_<tag>_<sel>.npy per
recording (existing files are skipped). With --example <recording>: also the inputs of the
slide figures for that recording (outputs file, consistency table, ring-model weights W).

  conda run -n parker python pull_local_inputs.py [tag sel] [--example <recording>]
"""
import glob
import os
import sys
import numpy as np
import h5py

src = "/mnt/kennedy_labdata/Parkerlab/results/raw_pipeline"
dst = "/home/rgast/data/parker_data/raw_pipeline"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
example = sys.argv[sys.argv.index("--example") + 1] if "--example" in sys.argv else None
if example:
    args.remove(example)
tag, sel = (args[0], args[1]) if len(args) > 1 else \
    ("p1_c0.6_pnr5_nrg0.999", "snr2_sz_rim5_hyb0.3-0.2_ves3_off6_fedge1.05")


def main():
    for w in sorted(glob.glob(f"{src}/*_rec?")):
        rec = os.path.basename(w)
        out, kept = f"{dst}/{rec}/lens_inputs_{tag}.npz", f"{dst}/{rec}/kept_{tag}_{sel}.npy"
        npz, fit = f"{w}/new_pipeline_outputs_{tag}_{sel}.npz", f"{w}/cnmfe_{tag}.hdf5"
        if (os.path.isfile(out) and os.path.isfile(kept)) or not (os.path.isfile(npz) and os.path.isfile(fit)):
            continue
        with h5py.File(fit, "r") as f:
            e = f["estimates"]
            d = {f"A_{k}": e["A"][k][()] for k in ("data", "indices", "indptr", "shape")}
            d.update(b0=e["b0"][()], sn=e["sn"][()], snr=e["SNR_comp"][()],
                     dims=f["dims"][()] if "dims" in f else np.array([192, 192]))
        with np.load(npz) as z:
            d.update(kept=z["component"], centroids_px=z["centroids_px"])
        os.makedirs(f"{dst}/{rec}", exist_ok=True)
        np.savez_compressed(out, **d)
        np.save(kept, d["kept"])
        print(rec, len(d["kept"]), flush=True)


def pull_example(rec):
    import shutil
    w, d = f"{src}/{rec}", f"{dst}/{rec}"
    for f in (f"new_pipeline_outputs_{tag}_{sel}.npz", f"eta_{tag}.csv"):
        if not os.path.isfile(f"{d}/{f}"):
            shutil.copy(f"{w}/{f}", f"{d}/{f}")
    out = f"{d}/cnmfe_extra_{tag}.npz"
    if not os.path.isfile(out):
        with h5py.File(f"{w}/cnmfe_{tag}.hdf5", "r") as f:
            e = f["estimates"]
            x = {f"W_{k}": e["W"][k][()] for k in ("data", "indices", "indptr", "shape")}
            x.update(snr=e["SNR_comp"][()], r_values=e["r_values"][()],
                     idx_components=e["idx_components"][()])
        np.savez_compressed(out, **x)
    print("example inputs:", rec)


if __name__ == "__main__":
    main()
    if example:
        pull_example(example)
