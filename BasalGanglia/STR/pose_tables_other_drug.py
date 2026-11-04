"""Pose-imaging alignment and arena tables for the sessions of another drug (no videos, no old
5 Hz speed traces), so that analysis_pose_arhmm.py can compute their features:

  offsets   pose_imaging_offsets_<drug>.csv: the fixed video offset of build_spn_dataset.py
            (imaging frame 0 = video frame 12007; checked on clozapine with the motion-correction
            shifts) and the imaging length from the recording's event matrix
            (old_frames_5hz = ceil(frames / 4), as the column is used for the feature length)
  arena     arena_circles_<drug>.csv: the circle of the same mouse's nearest clozapine video
            session (arena_circles.py; the drugs' recording days interleave); where the mouse
            explored the arena (keypoint-extent radius within 5 % of it), the session's own
            keypoint-extent centre (the camera can shift by up to ~20 px between days)

  conda run -n parker python pose_tables_other_drug.py Haloperidol
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

from arena_circles import extent_circle
from build_spn_dataset import video_offset

drug = sys.argv[1] if len(sys.argv) > 1 else "Haloperidol"
pose_root = f"/home/rgast/data/parker_data/pose_data/{drug}"
local = "/home/rgast/data/parker_data/raw_pipeline"
align_dir = "/home/rgast/data/parker_data/pose_alignment"
spikes = "spikes_p1_c0.6_pnr5_nrg0.999_thr1.5_sum3_snr2_sz_rim5_hyb0.3-0.2_ves3_off6_fedge1.05.npy"


def main():
    ref = pd.read_csv(f"{align_dir}/arena_circles.csv")
    vid = ref[ref.source == "video"].copy()
    vid["dt"] = pd.to_datetime(vid.date.astype(str))
    offs, arenas = [], []
    for f in sorted(glob.glob(f"{pose_root}/*/*/*_pose_top_v1_8.h5")):
        dose, part = f.split("/")[-3:-1]
        base = os.path.basename(f).split("_pose_top")[0]
        key = f"{drug.lower()}/{dose}/{base}"
        day = base if base.endswith("_amph") else base + "_amph"
        rec = f"{local}/{day}_rec{1 if part == 'Amph' else 0}/{spikes}"
        if not os.path.isfile(rec):
            print(f"no imaging for {key}")
            continue
        T = np.load(rec, mmap_mode="r").shape[1]
        offs.append(dict(session=key, dose=dose, part=part, old_frames_5hz=int(np.ceil(T / 4)),
                         offset_frames=video_offset, offset_s=video_offset / 20.0, fallback=False,
                         source="fixed video offset"))
        mouse = base.split("_")[1].replace("m85", "m085")
        ex, ey, er = extent_circle(f)
        same = vid[vid.mouse == mouse]
        row = dict(session=key, mouse=mouse, date=base[:8], ex=ex, ey=ey, eR=er)
        if len(same):
            j = (same.dt - pd.to_datetime(base[:8])).abs().idxmin()
            row.update(cx=same.cx[j], cy=same.cy[j], R_px=same.R_px[j], bx=same.cx[j], by=same.cy[j],
                       source=f"borrowed:{same.session[j].split('/')[-1]}")
            if abs(er / same.R_px[j] - 1) < 0.05:
                row.update(cx=ex, cy=ey, source=row["source"] + "+extent_centre")
        else:
            row.update(cx=ex, cy=ey, R_px=er, source="extent")
        arenas.append(row)
    o, a = pd.DataFrame(offs), pd.DataFrame(arenas)
    a["R_extent_rel"] = a.eR / a.R_px
    o.to_csv(f"{align_dir}/pose_imaging_offsets_{drug.lower()}.csv", index=False)
    a.to_csv(f"{align_dir}/arena_circles_{drug.lower()}.csv", index=False)
    print(f"{len(o)} sessions; arena sources: {a.source.str.split(':').str[0].value_counts().to_dict()}, "
          f"own extent centre: {a.source.str.endswith('extent_centre').sum()}; "
          f"extent / arena radius median {a.R_extent_rel.median():.2f}")


if __name__ == "__main__":
    main()
