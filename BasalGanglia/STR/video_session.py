"""Slide video: behaviour and SPN activity of one session (playback speed_factor x real time).

Top: arena video (top view) with the MARS keypoint skeleton | motion-corrected calcium
movie with the outlines of the extracted cells, each filled in proportion to its denoised
trace. Bottom: torso speed, AR-HMM state sequence and the population event rate (mean
over cells), with a cursor. Pose and imaging frames are aligned with the pose-imaging
offset (both 20 Hz; imaging frame i = video frame video_offset + i, see build_spn_dataset).

  conda run -n parker python video_session.py [t0_s duration_s]
Writes <SLIDE_DIR>/../videos/session.mp4 and a poster frame session_poster.png.
"""
import os
import sys
import numpy as np
import pandas as pd

import figures_pose_arhmm as F
import figures_raw_pipeline as R
from figures_pose_arhmm import A, STATE_COLORS, STATE_LABELS

out_dir = os.environ.get("VIDEO_DIR", "/home/rgast/OneDrive/presentations/parker/videos")
pose_session = "20191205_m085_clo01_amph"
win0 = 6800                          # start of the 120-s example window (frames)
fps = 20                             # recording frame rate (pose and imaging)
speed_factor = 0.5                   # playback speed (0.5: half speed, every frame shown)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.collections import PolyCollection
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
INK, INK2, MUTED, AXIS = R.INK, R.INK2, R.MUTED, R.AXIS

def arena_frames(off, t0, n):
    """Grayscale arena-video frames off + t0 ... (read sequentially, cached locally)."""
    cache = f"{R.local}/{R.rec}/arena_frames_{t0}_{t0 + n}.npy"
    if os.path.isfile(cache):
        return np.load(cache)
    import cv2
    cap = cv2.VideoCapture(f"{F.video_root}/{pose_session}_Top.avi")
    cap.set(cv2.CAP_PROP_POS_FRAMES, off + t0)
    out = []
    for _ in range(n):
        ok, img = cap.read()
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cap.release()
    out = np.stack(out)
    np.save(cache, out)
    return out

def cell_polygons(r, frac=0.4):
    """Outline polygon (x, y) of each kept cell at frac x its footprint maximum."""
    from skimage.measure import find_contours
    polys = []
    for k in r.kept:
        a = r.footprint(k)
        c = max(find_contours(a, frac * a.max()), key=len)
        polys.append(c[:, ::-1])
    return polys

def main(t0_s=0.0, dur_s=40.0):
    from scipy.ndimage import gaussian_filter, uniform_filter1d, gaussian_filter1d
    t0 = win0 + int(t0_s * fps)
    n = int(dur_s * fps)
    os.makedirs(out_dir, exist_ok=True)
    # --- data
    r = R.Recording(R.rec)
    Y = r.frames(win0, win0 + 2400)[t0 - win0:t0 - win0 + n]
    Y = gaussian_filter(uniform_filter1d(Y, 3, axis=0), (0, 1, 1))     # display denoising
    lo, hi = np.percentile(Y, [1, 99.7])
    C = r.C[:, t0:t0 + n]
    scale = np.percentile(r.C[:, win0:win0 + 2400], 99.5, axis=1)[:, None]
    act = np.clip(C / np.maximum(scale, 1e-9), 0, 1)
    rate = gaussian_filter1d(r.events[:, t0:t0 + n].mean(0).astype(float) * fps, 0.25 * fps)
    kp, centre, sc, off, ix = F.keypoints(pose_session)
    px = kp / sc
    from build_spn_dataset import video_offset
    tp = t0 + video_offset - off                 # pose feature row of imaging frame t0
    video = arena_frames(off, tp, n)
    feat = pd.read_csv(f"{F.A.out_dir}/features_{pose_session}.csv.gz", usecols=["speed"])
    speed = feat.speed.values[tp:tp + n]
    states = pd.read_csv(f"{F.A.out_dir}/pose_states.csv", index_col="state")
    # behaviour labels (merged fitted states, analysis_pose_arhmm.label_groups)
    lmap, names = F.label_map(states)
    z = lmap[pd.read_csv(f"{F.A.out_dir}/states_{pose_session}.csv.gz").state.values[tp:tp + n]]
    LCOL, LNAME = F.LABEL_COLORS, F.LABEL_NAMES
    # arena crop: square around the arena centre (keypoint extent)
    c = centre / sc
    H, W = video.shape[1:]
    half = H / 2
    x0 = int(np.clip(c[0] - half, 0, W - 2 * half))
    # --- figure (1280 x 720)
    fig = plt.figure(figsize=(8, 4.5), dpi=160)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.75, 1], hspace=0.12, left=0.11, right=0.875,
                          top=0.93, bottom=0.1)
    top = gs[0].subgridspec(1, 2, wspace=0.02)
    axv = fig.add_subplot(top[0])
    im_v = axv.imshow(video[0], cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    axv.set_xlim(x0, x0 + 2 * half)
    axv.set_ylim(H, 0)
    R.frame_axes(axv, "behaviour: top view + keypoints")
    sk = [axv.plot([], [], color="#f4f3ee", lw=1.4, solid_capstyle="round")[0] for _ in F.EDGES]
    dots = axv.scatter(np.zeros(7), np.zeros(7), s=10, color="#f4f3ee", edgecolor=INK, lw=0.4,
                       zorder=4)
    trail, = axv.plot([], [], color=R.ACCENT, lw=1.0, alpha=0.8)
    # current behavioural state in the arena centre (keypoint extent)
    arena_txt = axv.text(c[0], c[1], "", ha="center", va="center", fontsize=13,
                         fontweight="bold", zorder=6,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
    axc = fig.add_subplot(top[1])
    im_c = axc.imshow(Y[0], cmap=R.FLUO, vmin=lo, vmax=hi, interpolation="bilinear")
    R.frame_axes(axc, f"striatum: calcium imaging, {len(r.kept)} SPNs")
    polys = cell_polygons(r)
    base = np.array(to_rgba(R.ACCENT))
    pc = PolyCollection(polys, facecolors=np.zeros((len(polys), 4)),
                        edgecolors=[(1, 1, 1, 0.35)] * len(polys), linewidths=0.4)
    axc.add_collection(pc)
    R.scalebar(axc, r.dims)
    # time series
    sub = gs[1].subgridspec(3, 1, height_ratios=[1, 0.32, 1], hspace=0.18)
    t = np.arange(n) / fps
    ax1 = fig.add_subplot(sub[0])
    ax1.plot(t, speed, color=INK2, lw=0.8)
    ax1.set_ylabel("speed\n(cm/s)", rotation=0, ha="right", va="center")
    ax2 = fig.add_subplot(sub[1], sharex=ax1)
    ax2.imshow(z[None], cmap=ListedColormap([LCOL[k] for k in names]), vmin=-0.5,
               vmax=len(names) - 0.5, aspect="auto", extent=(0, t[-1], 0, 1),
               interpolation="nearest")
    ax2.set_yticks([])
    ax2.set_ylabel("behaviour", rotation=0, ha="right", va="center")
    ax3 = fig.add_subplot(sub[2], sharex=ax1)
    ax3.plot(t, rate, color=R.ACCENT, lw=1.0)
    ax3.set_ylabel("SPN rate\n(Hz)", rotation=0, ha="right", va="center")
    ax3.set_xlabel("time (s)")
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, t[-1])
    for ax in (ax1, ax3):
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelbottom=False, length=0)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    cursors = [ax.axvline(0, color=INK, lw=0.8) for ax in (ax1, ax2, ax3)]
    state_txt = ax2.text(1.01, 0.5, "", transform=ax2.transAxes, va="center", ha="left",
                         fontsize=8.5, fontweight="bold")
    pace = "real time" if speed_factor == 1 else f"{speed_factor:g}× speed"
    fig.text(0.99, 0.015, f"m085, vehicle + amphetamine · {pace}", ha="right", va="bottom",
             fontsize=7, color=MUTED)

    def draw(i):
        im_v.set_data(video[i])
        p = px[tp + i]
        for line, (a, b) in zip(sk, F.EDGES):
            line.set_data(*np.c_[p[ix[a]], p[ix[b]]])
        dots.set_offsets(p)
        tr = px[max(tp, tp + i - 20):tp + i + 1, ix["nose"]]
        trail.set_data(tr[:, 0], tr[:, 1])
        im_c.set_data(Y[i])
        fc = np.tile(base, (len(polys), 1))
        fc[:, 3] = 0.85 * act[:, i]
        pc.set_facecolors(fc)
        for cl in cursors:
            cl.set_xdata([t[i], t[i]])
        state_txt.set_text(LNAME[names[z[i]]])
        state_txt.set_color(LCOL[names[z[i]]])
        arena_txt.set_text(LNAME[names[z[i]]])
        arena_txt.set_color(LCOL[names[z[i]]])
        return []

    from matplotlib.animation import FuncAnimation, FFMpegWriter
    # poster: the frame with the highest population activity
    draw(int(np.argmax(rate)))
    fig.savefig(f"{out_dir}/session_poster.png", dpi=160)
    anim = FuncAnimation(fig, draw, frames=n, blit=False)
    writer = FFMpegWriter(fps=fps * speed_factor, codec="libx264", bitrate=4000,
                          extra_args=["-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "20"])
    anim.save(f"{out_dir}/session.mp4", writer=writer, dpi=160)
    plt.close(fig)
    print(f"saved {out_dir}/session.mp4 ({n} frames, {dur_s:g} s of recording, "
          f"{dur_s / speed_factor:g} s video) and session_poster.png")


if __name__ == "__main__":
    main(*(float(a) for a in sys.argv[1:3]))
