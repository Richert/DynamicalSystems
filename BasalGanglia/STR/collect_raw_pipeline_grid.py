"""Collect the CNMF-E seeding grid (min_corr x min_pnr) of pipeline_raw_spikes.py runs
into one table (per recording and averaged over recordings), incl. the stricter
post-filter (filt2) and the correlation-image peaks without an accepted cell.

  python collect_raw_pipeline_grid.py [work_root] [spike_thr] [rec1,rec2,...]
"""
import glob
import json
import os
import sys
import pandas as pd

work_root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "WORK_ROOT", "/home/rgast/data/parker_data/raw_pipeline")
thr = sys.argv[2] if len(sys.argv) > 2 else "1.5"

rows = []
for f in glob.glob(f"{work_root}/*_rec?/summary_p*_c*_pnr*_thr{thr}*.json"):
    s = json.load(open(f))
    rows.append(dict(recording=os.path.basename(os.path.dirname(f)),
                     min_corr=s["min_corr"], min_pnr=s["min_pnr"], filt2=s.get("filt2", False),
                     min_snr=s.get("min_snr", 3.0), min_eta=s.get("min_eta") or 0.0,
                     eta_dropped=s.get("n_eta_dropped", 0),
                     corr_peaks_uncovered=s.get("n_corr_peaks_uncovered", float("nan")),
                     corr_peaks=s.get("n_corr_peaks", float("nan")),
                     n_old=s["n_old"], n_new=s["n_new"],
                     active_old_matched=s["frac_active_old_matched"],   # share of old neurons with >=5 events
                     r_matched=s["r_matched_median"],
                     new_unmatched=s["n_new_unmatched"],
                     recall=s["recall_mean"], loss_s=s["loss_median"],
                     rate_hz=s["rate_new_median_hz"], noise_fp=s["noise_fp_share"]))
sel = ["min_corr", "min_pnr", "filt2", "min_snr", "min_eta"]
df = pd.DataFrame(rows).sort_values(["recording"] + sel)
if len(sys.argv) > 3:                       # restrict to the given recordings (comma-separated)
    df = df[df.recording.isin(sys.argv[3].split(","))]
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
print("\nmean over recordings:")
print(df.groupby(sel).mean(numeric_only=True).round(3).to_string())
out = f"{work_root}/seeding_grid_summary_thr{thr}.csv"
df.to_csv(out, index=False)
print(f"\nsaved {out}")
