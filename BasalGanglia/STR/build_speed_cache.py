"""Collect the old pipeline's 5 Hz speed traces of one drug / dose into a small h5 file
(datasets <drug>/<dose>/<session>, as in speed_traces_5hz.h5), so that the speed is
available without re-reading the large .mat files (which hold all traces). Reads one
file at a time (memory: one .mat file). Merge the output into the local cache with
  python build_speed_cache.py merge <out_file> [<cache>]

  python build_speed_cache.py <drug> <dose> <out_file>      (e.g. clozapine LowDose ...)
Path of the old pipeline output: NEURAL_ROOT (default: the server path).
"""
import os
import sys
import numpy as np
import h5py

neural_root = os.environ.get("NEURAL_ROOT", "/media/storage/DATA/Parkerlab/neural_data")
local_cache = "/home/rgast/data/parker_data/speed_traces_5hz.h5"
speed_field = "speed_traces_5hz"

def build(drug, dose, out):
    from scipy.io import loadmat
    with h5py.File(out, "a") as h:
        for session in sorted(os.listdir(f"{neural_root}/{drug}/{dose}")):
            cond = "amph" if session.endswith("_amph") else "veh"
            mat = f"{neural_root}/{drug}/{dose}/{session}/{cond}_drug.mat"
            key = f"{drug}/{dose}/{session}"
            if key in h or not os.path.isfile(mat):
                continue
            speed = np.asarray(loadmat(mat, simplify_cells=True)[f"{cond}_drug"][speed_field],
                               dtype=float)
            h.create_dataset(key, data=speed)
            h[key].attrs["source"] = mat
            print(f"{key}: {len(speed)} frames", flush=True)

def merge(src, dst=local_cache):
    with h5py.File(src, "r") as s, h5py.File(dst, "a") as d:
        keys = []
        s.visit(lambda k: keys.append(k) if isinstance(s[k], h5py.Dataset) else None)
        new = [k for k in keys if k not in d]
        for k in new:
            d.create_dataset(k, data=s[k][...])
            d[k].attrs.update(s[k].attrs)
    print(f"merged {len(new)} of {len(keys)} traces into {dst}")


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(*sys.argv[2:])
    else:
        build(*sys.argv[1:4])
