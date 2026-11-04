"""Check of the pose-imaging alignment against the motion-correction shifts: the frame-to-frame
brain motion (NoRMCorre rigid shifts, mc_shifts.npy) follows head / body movement, so its
cross-correlation with the keypoint speed (head speed + speed, both high-passed) peaks at lag 0
when the alignment is exact. Per recording: peak lag over +-L frames, and the peak in each
quarter of the recording (q_spread <= 4 frames: reliable). Used to find the fixed video offset
of build_spn_dataset.py (imaging frame 0 = video frame 12007).

  conda run -n parker python check_pose_imaging_alignment.py [out.csv]
"""
import sys
import os
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import median_filter
L = 40
def hp(x):
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    return x - median_filter(x, 21, mode='nearest')
def xc(a, b):
    a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
    return np.array([np.mean(a[:len(a) - k] * b[k:]) if k >= 0 else np.mean(a[-k:] * b[:k]) for k in range(-L, L + 1)])
rows, curves = [], []
with h5py.File('/home/rgast/data/parker_data/dataset/clozapine_spn.h5', 'r') as h:
    for day in h:
        for rec in ('rec0', 'rec1'):
            r = h[day][rec]; b = r['behaviour']
            if not b.attrs.get('available', False): continue
            sh = np.load(f'/home/rgast/data/parker_data/raw_pipeline/{day}_{rec}/mc_shifts.npy')
            m = hp(np.r_[0, np.linalg.norm(np.diff(sh, axis=0), axis=1)])
            v = hp(b['head_speed'][:] + b['speed'][:])
            n = min(len(v), len(m)); m, v = m[:n], v[:n]
            cc = xc(m, v); curves.append(cc)
            q = [int(np.argmax(xc(m[i*n//4:(i+1)*n//4], v[i*n//4:(i+1)*n//4])) - L) for i in range(4)]
            rows.append(dict(day=day, rec=rec, cond=r.attrs['condition'], lag=r.attrs['lag_frames'],
                             pose=r.attrs['pose_session'], peak=int(np.argmax(cc) - L), r=cc.max(), r0=cc[L],
                             q=q, q_spread=max(q) - min(q)))
d = pd.DataFrame(rows); C = np.array(curves)

d.to_csv(sys.argv[1] if len(sys.argv) > 1 else 'alignment_check.csv', index=False)
mc = C.mean(0)
print('mean curve peak', np.argmax(mc) - L, 'values -4..4:', np.round(mc[L-4:L+5], 3))
print(d[['peak','r','r0','q_spread']].describe().round(3))
print(d.peak.value_counts().sort_index().to_string())
rel = d[d.q_spread <= 2]
print('reliable (quarters within 2 frames):', len(rel), '\n', rel.peak.value_counts().sort_index().to_string())
print(d[(d.peak.abs() > 2)].sort_values('peak')[['day','rec','peak','r','q','q_spread']].to_string())
