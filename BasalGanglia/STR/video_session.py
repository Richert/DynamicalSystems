"""Slide video: behaviour and SPN activity of one session (playback speed_factor x real time).

Top: arena video (top view) with the MARS keypoint skeleton | motion-corrected calcium
movie with the outlines of the extracted cells, each filled in proportion to its denoised
trace. Bottom: torso speed, AR-HMM state sequence and the population event rate (mean
over cells), with a cursor. Pose and imaging frames are aligned with the pose-imaging
offset (both 20 Hz; imaging frame i = video frame video_offset + i, see build_spn_dataset).

  conda run -n parker python video_session.py [t0_s duration_s] [--columns]
  --columns: movies in a left column, behaviour feature traces + SPN rate on the right
  (session_features.mp4, first behaviour slide)
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

FEATURE_TRACES = [("speed", "speed\n(cm/s)", "torso"), ("turn", "turning\n(°/s)", "torso"),
                  ("head_angle", "head angle\n(°)", "head"), ("wall_dist", "wall dist.\n(cm)", "wall")]

def main_columns(t0_s=0.0, dur_s=40.0, name="session_features"):
    """Same snippet as main(): behaviour video (top) and calcium movie (bottom) in a left column;
    to the right, behaviour feature traces and the SPN population rate with a moving cursor."""
    from scipy.ndimage import gaussian_filter, uniform_filter1d, gaussian_filter1d
    t0 = win0 + int(t0_s * fps)
    n = int(dur_s * fps)
    os.makedirs(out_dir, exist_ok=True)
    r = R.Recording(R.rec)
    Y = r.frames(win0, win0 + 2400)[t0 - win0:t0 - win0 + n]
    Y = gaussian_filter(uniform_filter1d(Y, 3, axis=0), (0, 1, 1))
    lo, hi = np.percentile(Y, [1, 99.7])
    C = r.C[:, t0:t0 + n]
    scale = np.percentile(r.C[:, win0:win0 + 2400], 99.5, axis=1)[:, None]
    act = np.clip(C / np.maximum(scale, 1e-9), 0, 1)
    rate = gaussian_filter1d(r.events[:, t0:t0 + n].mean(0).astype(float) * fps, 0.25 * fps)
    kp, centre, sc, off, ix = F.keypoints(pose_session)
    px = kp / sc
    from build_spn_dataset import video_offset
    tp = t0 + video_offset - off
    video = arena_frames(off, tp, n)
    feat = pd.read_csv(f"{F.A.out_dir}/features_{pose_session}.csv.gz",
                       usecols=[c for c, _, _ in FEATURE_TRACES])
    c = centre / sc
    H, W = video.shape[1:]
    half = H / 2
    x0 = int(np.clip(c[0] - half, 0, W - 2 * half))
    fig = plt.figure(figsize=(8, 4.5), dpi=160)
    # left column: two square movies; right column: traces
    axv = fig.add_axes([0.015, 0.52, 0.255, 0.425])
    axc = fig.add_axes([0.015, 0.035, 0.255, 0.425])
    im_v = axv.imshow(video[0], cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    axv.set_xlim(x0, x0 + 2 * half)
    axv.set_ylim(H, 0)
    R.frame_axes(axv, "behaviour")
    sk = [axv.plot([], [], color="#f4f3ee", lw=1.4, solid_capstyle="round")[0] for _ in F.EDGES]
    dots = axv.scatter(np.zeros(7), np.zeros(7), s=10, color="#f4f3ee", edgecolor=INK, lw=0.4, zorder=4)
    trail, = axv.plot([], [], color=R.ACCENT, lw=1.0, alpha=0.8)
    im_c = axc.imshow(Y[0], cmap=R.FLUO, vmin=lo, vmax=hi, interpolation="bilinear")
    R.frame_axes(axc, f"striatum, {len(r.kept)} SPNs")
    polys = cell_polygons(r)
    base = np.array(to_rgba(R.ACCENT))
    pc = PolyCollection(polys, facecolors=np.zeros((len(polys), 4)),
                        edgecolors=[(1, 1, 1, 0.35)] * len(polys), linewidths=0.4)
    axc.add_collection(pc)
    R.scalebar(axc, r.dims)
    t = np.arange(n) / fps
    rows = FEATURE_TRACES + [("rate", "SPN rate\n(Hz)", None)]
    gs = fig.add_gridspec(len(rows), 1, left=0.42, right=0.985, top=0.96, bottom=0.1, hspace=0.25)
    axes, cursors = [], []
    for k, (col, lab, fc) in enumerate(rows):
        ax = fig.add_subplot(gs[k], sharex=axes[0] if axes else None)
        y = rate if col == "rate" else feat[col].values[tp:tp + n]
        ax.plot(t, y, color=R.ACCENT if col == "rate" else F.FC[fc], lw=1.0 if col == "rate" else 0.9)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        if col in ("turn", "head_angle"):                    # signed features: zero line
            ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
        ax.set_xlim(0, t[-1])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if k < len(rows) - 1:
            ax.tick_params(axis="x", labelbottom=False, length=0)
        else:
            ax.set_xlabel("time (s)")
        cursors.append(ax.axvline(0, color=INK, lw=0.8))
        axes.append(ax)
    pace = "real time" if speed_factor == 1 else f"{speed_factor:g}× speed"
    fig.text(0.99, 0.01, f"m085, vehicle + amphetamine · {pace}", ha="right", va="bottom",
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
        fc_ = np.tile(base, (len(polys), 1))
        fc_[:, 3] = 0.85 * act[:, i]
        pc.set_facecolors(fc_)
        for cl in cursors:
            cl.set_xdata([t[i], t[i]])
        return []

    from matplotlib.animation import FuncAnimation, FFMpegWriter
    draw(int(np.argmax(rate)))
    fig.savefig(f"{out_dir}/{name}_poster.png", dpi=160)
    anim = FuncAnimation(fig, draw, frames=n, blit=False)
    writer = FFMpegWriter(fps=fps * speed_factor, codec="libx264", bitrate=4000,
                          extra_args=["-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "20"])
    anim.save(f"{out_dir}/{name}.mp4", writer=writer, dpi=160)
    plt.close(fig)
    print(f"saved {out_dir}/{name}.mp4 and {name}_poster.png")



if __name__ == "__main__":
    if "--columns" in sys.argv:
        main_columns(*(float(a) for a in sys.argv[1:3] if a != "--columns"))
    else:
        main(*(float(a) for a in sys.argv[1:3]))


def title_banner(t_frame_s=24.0, win_s=40.0, n_traces=6,
                 out="/home/rgast/OneDrive/presentations/parker/logos/title_banner.png"):
    """Title-slide banner (~5:1): video frame with the keypoint skeleton | correlation image with the
    extracted cells | denoised traces and inferred spikes of a few cells (same session and window as
    the videos)."""
    from matplotlib.collections import PolyCollection
    t0 = win0
    n = int(win_s * fps)
    r = R.Recording(R.rec)
    kp, centre, sc, off, ix = F.keypoints(pose_session)
    px = kp / sc
    from build_spn_dataset import video_offset
    tp = t0 + video_offset - off
    i = int(t_frame_s * fps)
    img = F.video_frame(pose_session, off + tp + i)
    fig = plt.figure(figsize=(7.5, 1.5), dpi=300)
    # behaviour
    axv = fig.add_axes([0.0, 0.0, 0.2, 1.0])
    axv.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    c = centre / sc
    H, W = img.shape
    half = 0.5 * H
    x0 = int(np.clip(c[0] - half, 0, W - 2 * half))
    axv.set_xlim(x0, x0 + 2 * half)
    axv.set_ylim(H, 0)
    p = px[tp + i]
    for a, b in F.EDGES:
        axv.plot(*np.c_[p[ix[a]], p[ix[b]]], color="#f4f3ee", lw=1.2, solid_capstyle="round")
    axv.scatter(p[:, 0], p[:, 1], s=7, color=R.ACCENT, edgecolor=INK, lw=0.3, zorder=4)
    axv.axis("off")
    # imaging
    axc = fig.add_axes([0.205, 0.0, 0.2, 1.0])
    axc.imshow(r.cn, cmap=R.FLUO, vmin=np.percentile(r.cn, 1), vmax=1, interpolation="bilinear")
    polys = cell_polygons(r)
    from scipy.signal import find_peaks
    Cn = r.C[:, t0:t0 + n] / r.sn[:, None]
    npk, amp = np.zeros(len(Cn)), np.zeros(len(Cn))
    for k in range(len(Cn)):                       # cells with several clear transients of similar size
        pk, pr = find_peaks(Cn[k], prominence=6, distance=int(1.5 * fps))
        npk[k] = len(pk)
        amp[k] = np.median(pr["prominences"]) if len(pk) else 0
    cand = np.flatnonzero((npk >= 3) & (npk <= 8))
    pick = list(cand[np.argsort(-amp[cand])][:n_traces])
    cols = plt.get_cmap("plasma")(np.linspace(0.15, 0.85, n_traces))
    axc.add_collection(PolyCollection(polys, facecolors="none", edgecolors=[(1, 1, 1, 0.35)], linewidths=0.3))
    axc.add_collection(PolyCollection([polys[k] for k in pick], facecolors=[(*c_[:3], 0.55) for c_ in cols],
                                      edgecolors=cols, linewidths=0.8))
    axc.set_xlim(0, r.dims[1])
    axc.set_ylim(r.dims[0], 0)
    axc.axis("off")
    # traces
    axt = fig.add_axes([0.43, 0.04, 0.565, 0.92])
    t = np.arange(n) / fps
    for j, (k, c_) in enumerate(zip(pick, cols)):
        y = r.C[k, t0:t0 + n]
        y = np.clip(y / max(np.percentile(y, 99.5), 1e-9), 0, 1.15)
        off_y = -1.15 * j
        axt.plot(t, off_y + y, color=c_, lw=0.9)
        ts = t[r.events[k, t0:t0 + n]]
        axt.plot(ts, np.full(len(ts), off_y - 0.12), "|", color=c_, ms=3, mew=0.9)
    axt.set_xlim(0, t[-1])
    axt.axis("off")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")
