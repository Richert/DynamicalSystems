"""Arena outline (circle, video pixels) for every pose session.

The keypoint extent (used before) underestimates the arena when the mouse explores little
(sedated sessions: down to 0.6 x) and overestimates it with keypoint outliers (up to
1.14 x). Instead:
  video     circle fitted (Hough, radius 170-300 px) to the edges of the median of 5 frames
            of the session's top-view video: the arena wall is the dominant circular edge.
            On the 80 sessions with video, radius 224 or 240 px (two camera zoom settings),
            centre and radius stable within a mouse across days (mostly +/- 2 px)
  borrowed  no video (LowDose): the arena of the same mouse's video session nearest in time
            (recording days interleaved with the video days); where the mouse explored the
            arena (keypoint-extent radius within 5% of the borrowed one), the centre of the
            keypoint extent (34 / 42 sessions; radius ratio 1.01, centre offset median 5 px)
Output: pose_alignment/arena_circles.csv (session, cx, cy, R_px, source).

  conda run -n parker python arena_circles.py
"""
import os
import glob
import numpy as np
import pandas as pd
import h5py

video_root = os.environ.get("VIDEO_ROOT", "/mnt/kennedy_labdata/Parkerlab/videos/Clozapine")
pose_root = os.environ.get("POSE_ROOT", "/home/rgast/data/parker_data/pose_data/Clozapine")
align_dir = "/home/rgast/data/parker_data/pose_alignment"
offsets_csv = f"{align_dir}/pose_imaging_offsets.csv"
out = f"{align_dir}/arena_circles.csv"
radii = np.arange(170, 300, 2)

def video_circle(vid, n_frames=5):
    import cv2
    from skimage.feature import canny
    from skimage.transform import hough_circle, hough_circle_peaks
    cap = cv2.VideoCapture(vid)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr = []
    for t in np.linspace(0.1, 0.9, n_frames) * n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, img = cap.read()
        if ok:
            fr.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float))
    cap.release()
    m = np.median(fr, 0)                                 # the moving mouse is removed
    e = canny(m, sigma=3, low_threshold=0.85, high_threshold=0.95, use_quantiles=True)
    h = hough_circle(e, radii)
    acc, cx, cy, r = hough_circle_peaks(h, radii, total_num_peaks=1)
    return float(cx[0]), float(cy[0]), float(r[0]), float(acc[0])

def extent_circle(f):
    with h5py.File(f, "r") as h:
        kp = np.transpose(h["tracks"][0], (2, 1, 0))
    lo = np.nanpercentile(kp.reshape(-1, 2), 0.01, 0)
    hi = np.nanpercentile(kp.reshape(-1, 2), 99.99, 0)
    c = (lo + hi) / 2
    return float(c[0]), float(c[1]), float(np.mean(hi - lo) / 2)

def main():
    o = pd.read_csv(offsets_csv)
    prev = pd.read_csv(out).set_index("session") if os.path.isfile(out) else None
    rows = []
    for _, r in o.iterrows():
        drug, dose, base = r.session.split("/")
        f = f"{pose_root}/{dose}/{r.part}/{base}_pose_top_v1_8.h5"
        ex, ey, er = extent_circle(f)
        row = dict(session=r.session, mouse=base.split("_")[1].replace("m85", "m085"),
                   date=base[:8], ex=ex, ey=ey, eR=er)
        vid = f"{video_root}/{dose}/{r.part}/{base}_Top.mp4"
        if prev is not None and r.session in prev.index and prev.loc[r.session, "source"] == "video":
            row.update(prev.loc[r.session, ["cx", "cy", "R_px", "acc"]].to_dict(), source="video")
        elif os.path.isfile(vid):
            cx, cy, R, acc = video_circle(vid)
            row.update(cx=cx, cy=cy, R_px=R, acc=acc, source="video")
            print(f"{r.session}: video circle ({cx:.0f}, {cy:.0f}) R {R:.0f} (extent R {er:.0f})")
        rows.append(row)
    d = pd.DataFrame(rows)
    vid = d[d.source == "video"]
    for i in d.index[d.source != "video"]:
        same = vid[vid.mouse == d.mouse[i]]
        if not len(same):
            d.loc[i, ["cx", "cy", "R_px"]] = [d.ex[i], d.ey[i], d.eR[i]]
            d.loc[i, "source"] = "extent"
            continue
        j = (pd.to_datetime(same.date) - pd.to_datetime(d.date[i])).abs().idxmin()
        d.loc[i, ["cx", "cy", "R_px"]] = vid.loc[j, ["cx", "cy", "R_px"]].values
        d.loc[i, "source"] = f"borrowed:{vid.session[j].split('/')[-1]}"
        d.loc[i, ["bx", "by"]] = vid.loc[j, ["cx", "cy"]].values
        # well explored (extent radius within 5% of the borrowed one): the session's own
        # extent centre (accurate to ~5 px on the video sessions; the camera position can
        # shift by up to ~20 px between days)
        if abs(d.eR[i] / d.R_px[i] - 1) < 0.05:
            d.loc[i, ["cx", "cy"]] = [d.ex[i], d.ey[i]]
            d.loc[i, "source"] += "+extent_centre"
    d["R_extent_rel"] = d.eR / d.R_px
    d["centre_offset_px"] = np.hypot(d.ex - d.cx, d.ey - d.cy)
    d["borrowed_offset_px"] = np.hypot(d.ex - d.bx, d.ey - d.by)
    d.to_csv(out, index=False)
    b = d[d.source.str.startswith("borrowed")]
    ok = (b.R_extent_rel - 1).abs() < 0.05
    print(f"\n{(d.source == 'video').sum()} video, {len(b)} borrowed, "
          f"{(d.source == 'extent').sum()} extent")
    print(f"borrowed vs keypoint extent (sessions with good exploration, {ok.sum()}/{len(b)}): "
          f"radius ratio median {b.R_extent_rel[ok].median():.3f}, borrowed centre offset "
          f"median {b.borrowed_offset_px[ok].median():.1f} px (max "
          f"{b.borrowed_offset_px[ok].max():.1f}) -> extent centre used there")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
