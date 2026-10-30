"""Summary figures of the keypoint AR-HMM (selected model of analysis_pose_arhmm.py).

  fig_pose_features    feature extraction: keypoints and posture / movement features on
                       video frames, an example of the feature time series and the feature
                       correlation matrix (example session)
  fig_pose_hmm         fit result: state feature loadings, state confusability (cross-
                       validated decoder), transitions, bout durations, occupancy per
                       condition, example speed trace with the state sequence
  fig_pose_states      one typical bout per state (egocentric keypoints: start / end pose,
                       intermediate poses and keypoint displacement)

Inputs: the per-session features and the exported state sequences in OUT_DIR
(analysis_pose_arhmm.py, then analysis_pose_arhmm.py export), the MARS keypoints
(POSE_ROOT) and the top-view video of the example session (VIDEO_ROOT).

  conda run -n parker python figures_pose_arhmm.py
"""
import os
import glob
import numpy as np
import pandas as pd

os.environ.setdefault("OUT_DIR", "/home/rgast/data/parker_data/pose_arhmm_v5")
import analysis_pose_arhmm as A

video_root = os.environ.get("VIDEO_ROOT", "/mnt/kennedy_labdata/Parkerlab/sample_videos")
fig_dir = f"{A.out_dir}/figures"
example = "20191205_m085_clo01_amph"      # example session (amphetamine, video available)
window_s = 120                             # example time series (fig. 1; fig. 2: first 60 s)
seed = 1

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch

# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "figure.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "none"})

# states: hue = behaviour family (blue immobile, orange locomotion, violet turning, aqua
# excluded rear / artefact), lightness = order within the family; names are shown wherever colour is used
STATE_LABELS = {"rest": "rest", "quiet": "quiet", "shift": "shift in place",
                "slow_walk": "slow walk", "reorient": "reorient", "walk": "walk",
                "rear_artifact": "rear / artefact", "turn_fast": "fast turn", "run": "run",
                "run_fast": "fast run"}
STATE_COLORS = {"rest": "#0d366b", "quiet": "#2a78d6", "shift": "#86b6ef",
                "slow_walk": "#f3a77f", "walk": "#eb6834", "run": "#b84a1c",
                "run_fast": "#6e2a0e", "reorient": "#a99ef0", "turn_fast": "#5140c4",
                "rear_artifact": "#1baf7a"}
START_GREY = "#d3d1cb"                     # bout start in fig 3
# feature annotation colours (fig 1), shared between the frames and the time series
FC = dict(torso="#2a78d6", head="#eb6834", bend="#6250d6", length="#c98500", ears="#1baf7a",
          hips="#e87ba4", wall="#008300")
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec",
                                                      "#ec8a85", "#a8302f"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#f7f9fc", "#b7d3f6", "#5598e7",
                                                       "#1c5cab", "#0d366b"])
FEAT_LABELS = {"log_speed": "speed (log)", "turn_asinh": "turn", "head_angle": "head angle",
               "body_len": "body length", "bend": "body bend", "log_head_speed": "head speed (log)",
               "head_turn_asinh": "head turn", "ear_width": "ear width", "hip_width": "hip width"}

def panel(ax, letter, x=-0.12, y=1.04, fx=None):
    """Panel letter at axes coordinates (x, y), or at figure x = fx (aligned letters)."""
    from matplotlib.transforms import blended_transform_factory
    tr = ax.transAxes if fx is None else blended_transform_factory(ax.figure.transFigure,
                                                                   ax.transAxes)
    ax.text(x if fx is None else fx, y, letter, transform=tr, fontsize=10, fontweight="bold",
            color=INK, va="bottom", ha="left")

def tag_in(ax, xy, text, color, ha="left", margin=0.12):
    """tag() with the anchor kept inside the axes limits."""
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    mx, my = margin * (x1 - x0), 0.06 * (y1 - y0)
    if ha == "left":
        x = np.clip(xy[0], x0 + 0.02 * (x1 - x0), x1 - 3 * mx)
    else:
        x = np.clip(xy[0], x0 + 1.5 * mx, x1 - 1.5 * mx)
    tag(ax, (x, np.clip(xy[1], y0 + my, y1 - my)), text, color, ha=ha)

def save(fig, name):
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{fig_dir}/{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"saved {fig_dir}/{name}.png/.svg")

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def pose_file(session):
    f = glob.glob(f"{A.pose_root}/*/*/{session}_pose_top_v1_8.h5")[0]
    dose, part = f.split("/")[-3:-1]
    return f, dose, part

def keypoints(session):
    """Smoothed keypoints of the imaging period (cm), arena centre (cm), cm/px, offset."""
    f, dose, _ = pose_file(session)
    o = pd.read_csv(A.offsets_csv).set_index("session").loc[f"{A.drug.lower()}/{dose}/{session}"]
    off, scale = int(o.offset_frames), float(o.cm_per_px)
    kp, _, centre, nodes = A.load_keypoints(f, off, int(o.old_frames_5hz) * 4, scale)
    return kp, centre, scale, off, {n: i for i, n in enumerate(nodes)}

def video_frame(session, frame):
    import cv2
    cap = cv2.VideoCapture(f"{video_root}/{session}_Top.avi")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if ok else None

def load_data():
    states = pd.read_csv(f"{A.out_dir}/pose_states.csv", index_col="state")
    sessions = []
    for f in sorted(glob.glob(f"{A.out_dir}/features_*.csv.gz")):
        name = os.path.basename(f)[9:-7]
        z = pd.read_csv(f"{A.out_dir}/states_{name}.csv.gz").state.values
        feat = pd.read_csv(f).iloc[:len(z)].reset_index(drop=True)
        _, dose, part = pose_file(name)
        sessions.append(dict(session=name, condition=A.cond_of[(dose, part)], feat=feat, z=z))
    allf = pd.concat([s["feat"][A.features] for s in sessions])
    mu, sd = allf.mean(), allf.std()
    for s in sessions:
        s["X"] = ((s["feat"][A.features] - mu) / sd).values
    return states, sessions

def bouts(sessions):
    """All state bouts: session index, start, end (frames), state, mean z-features,
    mean interpolated-keypoint share."""
    rows = []
    for i, s in enumerate(sessions):
        z = s["z"]
        st = np.flatnonzero(np.diff(np.r_[-1, z]) != 0)
        en = np.r_[st[1:], len(z)]
        mx = np.add.reduceat(s["X"], st, axis=0) / (en - st)[:, None]
        ish = np.add.reduceat(s["feat"].interp_share.values, st) / (en - st)
        d = pd.DataFrame(mx, columns=A.features)
        d.insert(0, "interp", ish)
        d.insert(0, "state", z[st])
        d.insert(0, "end", en)
        d.insert(0, "start", st)
        d.insert(0, "sess", i)
        rows.append(d)
    b = pd.concat(rows, ignore_index=True)
    b["dur"] = (b.end - b.start) / A.fs
    return b

def example_window(s, states):
    """Window of the example session with the most even state mix (excluded states and
    frames with interpolated keypoints avoided)."""
    n, w = len(s["z"]), int(window_s * A.fs)
    bad = set(states.index[states.excluded])
    best, t_best = -1, 0
    for t0 in range(0, n - w, int(5 * A.fs)):
        z = s["z"][t0:t0 + w]
        if np.isin(z, list(bad)).any() or s["feat"].interp_share.values[t0:t0 + w].mean() > 0.02:
            continue
        p = np.bincount(z, minlength=len(states)) / w
        h = -(p[p > 0] * np.log(p[p > 0])).sum()
        if h > best:
            best, t_best = h, t0
    return t_best, t_best + w

# ---------------------------------------------------------------------------
# drawing helpers (pixel coordinates of the video, y pointing down)
# ---------------------------------------------------------------------------

EDGES = [("nose", "ear1"), ("nose", "ear2"), ("ear1", "neck"), ("ear2", "neck"),
         ("neck", "hip1"), ("neck", "hip2"), ("hip1", "tail_base"), ("hip2", "tail_base")]

def skeleton(ax, p, ix, color="white", lw=1.0, ms=9, alpha=1.0, edge=INK, z=3):
    for a, b in EDGES:
        ax.plot(*np.c_[p[ix[a]], p[ix[b]]], color=color, lw=lw, alpha=alpha, zorder=z,
                solid_capstyle="round")
    ax.scatter(p[:, 0], p[:, 1], s=ms, color=color, edgecolor=edge, lw=0.5, alpha=alpha,
               zorder=z + 1)

def arc(ax, c, v0, v1, r, color, lw=1.2, fill=False):
    """Arc of radius r around c from direction v0 to v1 (optionally a filled wedge);
    returns a point just outside its middle."""
    a0 = np.arctan2(v0[1], v0[0])
    d = np.arctan2(v0[0] * v1[1] - v0[1] * v1[0], v0 @ v1)
    t = np.linspace(a0, a0 + d, 30)
    ax.plot(c[0] + r * np.cos(t), c[1] + r * np.sin(t), color=color, lw=lw, zorder=5)
    if fill:
        ax.fill(np.r_[c[0], c[0] + r * np.cos(t)], np.r_[c[1], c[1] + r * np.sin(t)],
                color=color, alpha=0.35, lw=0, zorder=5)
    return c + r * 1.35 * np.array([np.cos(a0 + d / 2), np.sin(a0 + d / 2)])

def arrow(ax, p0, p1, color, lw=1.2, ms=7, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, zorder=z, shrinkA=0, shrinkB=0))

def tag(ax, xy, text, color, ha="left", va="center"):
    ax.text(*xy, text, color=color, fontsize=6.5, ha=ha, va=va, zorder=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

def callouts(ax, centre, items, dist, min_sep=np.radians(38)):
    """Labels on a circle of radius dist around centre, each at the angle of its anchor
    (spread apart to at least min_sep), joined to the anchor by a leader line.
    items: (anchor, text, colour)."""
    ang = np.array([np.arctan2(*(a - centre)[::-1]) for a, _, _ in items])
    order = np.argsort(ang)
    a = ang[order].copy()
    for _ in range(200):                                     # relax neighbouring labels apart
        d = np.diff(np.r_[a, a[0] + 2 * np.pi])
        if d.min() >= min_sep - 1e-3:
            break
        i = int(np.argmin(d))
        j = (i + 1) % len(a)
        push = (min_sep - d[i]) / 2 + 1e-3
        a[i] -= push
        a[j] += push
    ang[order] = a
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    w = x1 - x0
    for (anchor, text, color), t in zip(items, ang):
        pos = centre + dist * np.array([np.cos(t), np.sin(t)])
        ha = "left" if np.cos(t) > 0.3 else "right" if np.cos(t) < -0.3 else "center"
        lo, hi = {"left": (x0, x1 - 0.45 * w), "right": (x0 + 0.45 * w, x1),
                  "center": (x0 + 0.16 * w, x1 - 0.16 * w)}[ha]
        pos = np.array([np.clip(pos[0], lo, hi), np.clip(pos[1], y0 + 0.05 * w, y1 - 0.05 * w)])
        ax.plot(*np.c_[anchor, pos], color=color, lw=0.5, zorder=6)
        tag(ax, pos, text, color, ha=ha)

def show_frame(ax, img):
    ax.imshow(img, cmap="gray", vmin=-70, vmax=255, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(AXIS)

# ---------------------------------------------------------------------------
# figure 1: feature extraction
# ---------------------------------------------------------------------------

def fig_features(states, sessions):
    s = next(x for x in sessions if x["session"] == example)
    feat = s["feat"]
    kp, centre, scale, off, ix = keypoints(example)
    px = kp / scale                                                # pixels
    t0, t1 = example_window(s, states)
    clean = feat.interp_share.rolling(11, center=True).max().values == 0
    # example frames: still with plausible keypoints (lengths / widths near the session
    # median), mouse well inside the image and away from the wall; (b) clearly bent body,
    # (c) clearly turned head (another frame), (d) moderate-speed fast turn
    win = np.zeros(len(feat), bool)
    win[t0 + 20:t1 - 20] = True
    tors_px = px[:, [ix["neck"], ix["hip1"], ix["hip2"], ix["tail_base"]]].mean(1)
    win &= (tors_px.min(1) > 90) & (tors_px[:, 0] < 640 - 90) & (tors_px[:, 1] < 480 - 90)
    normal = feat.body_len.between(0.95, 1.08) & feat.ear_width.between(0.95, 1.05) & \
        feat.hip_width.between(0.93, 1.07)
    still = win & clean & normal & (feat.speed < 0.5) & (feat.wall_dist > 4)
    if not still.any():
        still = win & clean & normal & (feat.speed < 0.5)
    tb = int(feat.bend.abs().where(still & feat.bend.abs().between(10, 35)
                                   & (feat.head_angle.abs() < 20)).idxmax())
    far = np.abs(np.arange(len(feat)) - tb) > 3 * A.fs
    tc = int(feat.head_angle.abs().where(still & far & feat.head_angle.abs().between(25, 55))
             .idxmax())
    moving = win & clean & np.isin(s["z"], [states.index[states.name == "turn_fast"][0]]) & \
        feat.speed.between(6, 14) & far & (np.abs(np.arange(len(feat)) - tc) > 3 * A.fs)
    td = int(feat.turn.abs().where(moving).idxmax())
    torso_of = lambda q: q[[ix["neck"], ix["hip1"], ix["hip2"], ix["tail_base"]]].mean(0)

    fig = plt.figure(figsize=(9.6, 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.55, 1.6], hspace=0.14)
    top = gs[0].subgridspec(1, 4, wspace=0.14, width_ratios=[1.33, 1, 1, 1])
    bot = gs[1].subgridspec(1, 2, wspace=0.32, width_ratios=[2.1, 1])

    def zoom(ax, q, z=80):
        H, W = 480, 640
        cz = np.clip(q, z, [W - z, H - z])
        ax.set_xlim(cz[0] - z, cz[0] + z)
        ax.set_ylim(cz[1] + z, cz[1] - z)
        return cz

    # (a) full frame: arena, wall distance
    ax = fig.add_subplot(top[0])
    img = video_frame(example, off + tb)
    show_frame(ax, img)
    p = px[tb]
    R = A.arena_cm / 2 / scale
    c = centre / scale
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(c[0] + R * np.cos(th), c[1] + R * np.sin(th), color=MUTED, lw=0.8, ls=(0, (3, 2)))
    torso = torso_of(p)
    u = (torso - c) / np.linalg.norm(torso - c)
    wall = c + R * u
    ax.plot(*np.c_[torso, wall], color=FC["wall"], lw=1.4, zorder=5)
    skeleton(ax, p, ix, ms=5, lw=0.7)
    H, W = img.shape
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    zb = 80
    zcen = np.clip(torso, zb, [W - zb, H - zb])                    # zoom box inside the frame
    ax.add_patch(Rectangle(zcen - zb, 2 * zb, 2 * zb, fill=False, ec=INK, lw=0.6, ls="--"))
    # wall-distance label beside the line, on the side away from the keypoints
    perp = np.array([-u[1], u[0]])
    wm = (torso + wall) / 2
    if ((p - wm) @ perp).mean() > 0:
        perp = -perp
    pos = wall + 0.6 * zb * perp
    ax.plot(*np.c_[wm, pos], color=FC["wall"], lw=0.5, zorder=6)
    tag_in(ax, pos, "wall distance", FC["wall"], ha="left" if perp[0] > 0 else "right")
    tag_in(ax, c + [0, 0.85 * R], "arena (keypoint extent)", INK2, ha="center", margin=0.2)
    ax.set_title("top-view frame with MARS keypoints", loc="left")
    panel(ax, "a", fx=0.01, y=1.02)

    # (b) body posture: body axis, body length, bend, hip width
    ax = fig.add_subplot(top[1])
    show_frame(ax, img)
    nose, neck, tail = p[ix["nose"]], p[ix["neck"]], p[ix["tail_base"]]
    hips = (p[ix["hip1"]] + p[ix["hip2"]]) / 2
    skeleton(ax, p, ix, ms=10, lw=0.8)
    zoom(ax, torso, zb)
    body = neck - tail
    arrow(ax, tail, neck, FC["torso"], lw=1.0)
    # bend: angle at the hip midpoint between the tail -> hips segment (extended, dashed)
    # and the hips -> neck segment
    seg1 = hips - tail
    ax.plot(*np.c_[tail, hips], color=FC["bend"], lw=2.0, zorder=6, solid_capstyle="round")
    ax.plot(*np.c_[hips, neck], color=FC["bend"], lw=2.0, zorder=6, solid_capstyle="round")
    ax.plot(*np.c_[hips, hips + 1.1 * seg1 / np.linalg.norm(seg1) * np.linalg.norm(neck - hips)],
            color=FC["bend"], lw=0.9, ls=(0, (2, 1.5)), zorder=6)
    ax.scatter(*hips, s=14, color=FC["bend"], edgecolor="white", lw=0.6, zorder=7)
    a_bend = arc(ax, hips, seg1, neck - hips, 38, FC["bend"], lw=1.2, fill=True)
    ax.plot(*np.c_[p[ix["hip1"]], p[ix["hip2"]]], color=FC["hips"], lw=1.6, zorder=5)
    n = np.array([-body[1], body[0]]) / np.linalg.norm(body)       # body normal
    side = 1 if (neck - hips) @ n < 0 else -1                      # opposite to the bend
    o = side * 24 * n
    ax.plot(*np.c_[tail + o, nose + o], color=FC["length"], lw=1.0, zorder=5)
    for q in (tail, nose):
        ax.plot(*np.c_[q + o * 0.75, q + o * 1.25], color=FC["length"], lw=1.0, zorder=5)
    mid = (tail + nose) / 2
    callouts(ax, mid, [
        (mid + o, "body length", FC["length"]),
        (a_bend, f"bend ({feat.bend[tb]:+.0f}°)".replace("-", "−"), FC["bend"]),
        (p[ix["hip1"]] * 0.35 + p[ix["hip2"]] * 0.65, "hip width", FC["hips"]),
        (tail + 0.35 * body, "body axis", FC["torso"])], dist=0.68 * zb,
        min_sep=np.radians(55))
    ax.set_title("body posture", loc="left")
    panel(ax, "b", x=-0.1, y=1.02)

    # (c) head posture: head angle, ear width (another still frame)
    ax = fig.add_subplot(top[2])
    show_frame(ax, video_frame(example, off + tc))
    q = px[tc]
    nose, neck, tail = q[ix["nose"]], q[ix["neck"]], q[ix["tail_base"]]
    skeleton(ax, q, ix, ms=10, lw=0.8)
    cz = zoom(ax, (torso_of(q) + nose) / 2, 70)
    body = neck - tail
    ax.plot(*np.c_[tail, neck], color=FC["torso"], lw=1.0, zorder=5)
    ax.plot(*np.c_[neck, neck + 0.5 * body], color=FC["torso"], lw=0.9, ls=(0, (2, 1.5)),
            zorder=5)
    arrow(ax, neck, nose, FC["head"], lw=1.6)
    a_head = arc(ax, neck, body, nose - neck, 24, FC["head"], lw=1.2, fill=True)
    ax.plot(*np.c_[q[ix["ear1"]], q[ix["ear2"]]], color=FC["ears"], lw=2.0, zorder=6)
    callouts(ax, neck, [
        (a_head, f"head angle ({feat.head_angle[tc]:+.0f}°)".replace("-", "−"), FC["head"]),
        ((q[ix["ear1"]] + q[ix["ear2"]]) / 2, "ear width", FC["ears"]),
        (tail * 0.4 + neck * 0.6, "body axis", FC["torso"])], dist=0.75 * 70,
        min_sep=np.radians(60))
    ax.set_title("head posture", loc="left")
    panel(ax, "c", x=-0.1, y=1.02)

    # (d) movement features: torso / nose paths, torso velocity, heading change
    ax = fig.add_subplot(top[3])
    img = video_frame(example, off + td)
    show_frame(ax, img)
    lag = int(0.4 * A.fs)
    seg = px[td - lag:td + 1]
    zoom(ax, (seg.reshape(-1, 2).min(0) + seg.reshape(-1, 2).max(0)) / 2, zb)
    tors = seg[:, [ix["neck"], ix["hip1"], ix["hip2"], ix["tail_base"]]].mean(1)
    skeleton(ax, seg[0], ix, color="white", ms=6, lw=0.6, alpha=0.6)
    skeleton(ax, seg[-1], ix, ms=10, lw=0.8)
    ax.plot(tors[:, 0], tors[:, 1], color=FC["torso"], lw=1.4, zorder=6)
    ax.plot(seg[:, ix["nose"], 0], seg[:, ix["nose"], 1], color=FC["head"], lw=1.4, zorder=6)
    b0 = seg[0, ix["neck"]] - seg[0, ix["tail_base"]]
    b1 = seg[-1, ix["neck"]] - seg[-1, ix["tail_base"]]
    v = (tors[-1] - tors[-3]) / 2 * A.fs * 0.25                   # px moved in 0.25 s
    arrow(ax, tors[-1], tors[-1] + v, FC["torso"], lw=1.4)
    tag_in(ax, tors[-1] + 1.3 * v, "speed", FC["torso"], ha="center")
    ta = arc(ax, tors[0], b0, b1, 24, FC["torso"])
    tag_in(ax, tors[0] + 2.2 * (ta - tors[0]), "turn", FC["torso"], ha="center")
    nz = seg[-1, ix["nose"]]
    tag_in(ax, nz + 22 * (nz - tors[-1]) / np.linalg.norm(nz - tors[-1]), "head speed / turn",
           FC["head"], ha="center", margin=0.3)
    ax.set_title(f"movement (last {lag / A.fs:.1f} s)", loc="left")
    panel(ax, "d", x=-0.1, y=1.02)

    # (e) feature time series
    rows = [("speed", "speed\ncm/s", FC["torso"]), ("turn", "turn\n°/s", FC["torso"]),
            ("head_angle", "head\nangle °", FC["head"]),
            ("head_speed", "head speed\ncm/s", FC["head"]),
            ("head_turn", "head\nturn °/s", FC["head"]), ("bend", "bend °", FC["bend"]),
            ("body_len", "body\nlength", FC["length"]), ("ear_width", "ear\nwidth", FC["ears"]),
            ("hip_width", "hip\nwidth", FC["hips"]), ("wall_dist", "wall\ndist. cm", FC["wall"])]
    sub = bot[0].subgridspec(len(rows), 1, hspace=0.18)
    t = (np.arange(t0, t1) - t0) / A.fs
    for r, (col, lab, color) in enumerate(rows):
        ax = fig.add_subplot(sub[r])
        y = feat[col].values[t0:t1]
        if col in ("turn", "head_turn"):
            lim = np.percentile(np.abs(y), 99)
            y = np.clip(y, -lim, lim)
        if col in ("turn", "head_angle", "head_turn", "bend"):
            ax.axhline(0, color=GRID, lw=0.6)
        ax.plot(t, y, color=color, lw=0.7)
        for tt, l in ((tb, "a, b"), (tc, "c"), (td, "d")):
            if t0 <= tt < t1:
                ax.axvline((tt - t0) / A.fs, color=MUTED, lw=0.5, ls=(0, (2, 2)))
                if r == 0:
                    ax.text((tt - t0) / A.fs, 1.08, l, transform=ax.get_xaxis_transform(),
                            ha="center", fontsize=6.5, fontweight="bold", color=INK2)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6.3, labelpad=4)
        ax.set_xlim(0, window_s)
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        if r < len(rows) - 1:
            ax.tick_params(axis="x", length=0, labelbottom=False)
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_xlabel("time (s)")
        if r == 0:
            ax.set_title(f"feature time series (example session, amphetamine)", loc="left", pad=10)
            panel(ax, "e", fx=0.01, y=1.3)

    # (f, g) correlation matrices of the HMM input features: example session and the mean
    # over all sessions
    right = bot[1].subgridspec(2, 1, hspace=0.55)
    labels = [FEAT_LABELS[f] for f in A.features]
    mats = [(np.corrcoef(s["X"].T), "feature correlations\n(HMM inputs, example session)"),
            (np.mean([np.corrcoef(x["X"].T) for x in sessions], 0),
             f"mean over all {len(sessions)} sessions")]
    for r, (C, title) in enumerate(mats):
        ax = fig.add_subplot(right[r])
        C = C.copy()
        C[np.eye(len(C), dtype=bool)] = np.nan
        ax.imshow(C, cmap=DIVERGING, vmin=-1, vmax=1)
        for i in range(len(C)):
            for j in range(len(C)):
                if i != j:
                    ax.text(j, i, f"{C[i, j]:.2f}".replace("0.", ".").replace("-.", "−."),
                            ha="center", va="center", fontsize=4.6,
                            color="white" if abs(C[i, j]) > 0.6 else INK2)
        ax.set_xticks(range(len(C)), labels if r else [], rotation=55, ha="right")
        ax.set_yticks(range(len(C)), labels)
        ax.tick_params(length=0, labelsize=6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc="left")
        panel(ax, "fg"[r], fx=0.645, y=1.02)
        if r == 0:
            sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-1, 1))
            cb = fig.colorbar(sm, cax=ax.inset_axes([1.05, 0.2, 0.05, 0.6]), ticks=[-1, 0, 1])
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=6)
            cb.set_label("Pearson r", color=INK2)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.96, bottom=0.09)
    save(fig, "fig_pose_features")

# ---------------------------------------------------------------------------
# figure 2: HMM fit
# ---------------------------------------------------------------------------

def decoder_confusion(sessions, K):
    """Cross-validated confusion of a Gaussian (QDA) decoder of the state from the
    features at t and their change over the AR context (0.2 s); sessions split in two
    halves (fit on one, test on the other, both ways), uniform state priors."""
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    lag = A.ar_lags
    def data(idx):
        X, y = [], []
        for i in idx:
            x = sessions[i]["X"]
            X.append(np.c_[x[lag:], x[lag:] - x[:-lag]][::5])
            y.append(sessions[i]["z"][lag:][::5])
        return np.concatenate(X), np.concatenate(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sessions))
    halves = perm[::2], perm[1::2]
    cm = np.zeros((K, K))
    for tr, te in (halves, halves[::-1]):
        X, y = data(tr)
        qda = QuadraticDiscriminantAnalysis(priors=np.full(K, 1 / K), reg_param=0.01).fit(X, y)
        Xt, yt = data(te)
        pred = qda.predict(Xt)
        np.add.at(cm, (yt, pred), 1)
    return cm / cm.sum(1, keepdims=True)

def heat(ax, M, labels_y, labels_x, cmap, vmin, vmax, fmt, thr, mask_diag=False):
    M = np.array(M, float)
    if mask_diag:
        M[np.eye(len(M), dtype=bool)] = np.nan
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]) and abs(M[i, j]) >= thr:
                ax.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=5.2,
                        color="white" if abs(M[i, j]) > 0.6 * max(abs(vmin), vmax) else INK2)
    ax.set_xticks(range(M.shape[1]), labels_x, rotation=50, ha="right")
    ax.set_yticks(range(M.shape[0]), labels_y)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

def state_ticklabels(ax, names, axis="y"):
    labs = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for l, n in zip(labs, names):
        l.set_color(STATE_COLORS[n])
        l.set_fontweight("bold")

def fig_hmm(states, sessions, b):
    K = len(states)
    names = states.name.tolist()
    labels = [STATE_LABELS[n] + ("*" if states.excluded[k] else "") for k, n in enumerate(names)]
    fig = plt.figure(figsize=(7.2, 9.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.95, 0.6], hspace=0.75)
    r1 = gs[0].subgridspec(1, 3, wspace=0.12, width_ratios=[1, 1, 1])
    r2 = gs[1].subgridspec(1, 2, wspace=0.5, width_ratios=[1.6, 1])

    # (a) loadings: mean z-scored HMM input feature per state
    ax = fig.add_subplot(r1[0])
    allX = np.concatenate([s["X"] for s in sessions])
    allz = np.concatenate([s["z"] for s in sessions])
    L = np.array([allX[allz == k].mean(0) for k in range(K)])
    heat(ax, L, labels, [FEAT_LABELS[f] for f in A.features], DIVERGING, -2, 2,
         lambda v: f"{v:.1f}".replace("-", "−"), 0.5)
    state_ticklabels(ax, names)
    ax.set_xticks(range(len(A.features)), [FEAT_LABELS[f] for f in A.features], rotation=90,
                  ha="center")
    ax.set_title("feature loadings\n(state mean, z-scored)", loc="left")
    panel(ax, "a", fx=0.01)

    # (b) decoder confusion
    ax = fig.add_subplot(r1[1])
    cm = decoder_confusion(sessions, K)
    heat(ax, cm, labels, labels, SEQUENTIAL, 0, 1, lambda v: f"{v:.2f}".lstrip("0"), 0.05)
    ax.set_xticks(range(K), labels, rotation=90, ha="center")
    state_ticklabels(ax, names, "x")
    ax.set_yticklabels([])
    ax.set_xlabel("decoded state")
    ax.set_ylabel("HMM state", labelpad=2)
    ax.set_title(f"state confusability (held-out\ndecoder; mean diagonal {np.diag(cm).mean():.2f})",
                 loc="left")
    panel(ax, "b", x=-0.12)

    # (c) transitions between states (switches only)
    ax = fig.add_subplot(r1[2])
    T = np.zeros((K, K))
    for s in sessions:
        z = s["z"]
        sw = np.flatnonzero(np.diff(z) != 0)
        np.add.at(T, (z[sw], z[sw + 1]), 1)
    T = T / T.sum(1, keepdims=True)
    heat(ax, T, labels, labels, SEQUENTIAL, 0, 1, lambda v: f"{v:.2f}".lstrip("0"), 0.05,
         mask_diag=True)
    ax.set_xticks(range(K), labels, rotation=90, ha="center")
    state_ticklabels(ax, names, "x")
    ax.set_yticklabels([])
    ax.set_xlabel("next state")
    ax.set_ylabel("current state", labelpad=2)
    ax.set_title("transitions\n(P(next state | switch))", loc="left")
    panel(ax, "c", x=-0.12)

    # (d) bout durations
    ax = fig.add_subplot(r2[0])
    for k in range(K):
        d = np.log10(b.dur[b.state == k].values)
        if len(d) > 5000:
            d = np.random.default_rng(seed).choice(d, 5000, replace=False)
        vp = ax.violinplot(d, [k], widths=0.8, showextrema=False, bw_method=0.35)
        for body in vp["bodies"]:
            body.set_facecolor(STATE_COLORS[names[k]])
            body.set_edgecolor("white")
            body.set_linewidth(0.8)
            body.set_alpha(0.9)
        med = np.median(b.dur[b.state == k])
        ax.plot([k - 0.18, k + 0.18], [np.log10(med)] * 2, color=INK, lw=1.2)
        ax.text(k, np.log10(med) + 0.07, f"{med:.2f}", ha="center", fontsize=5.5, color=INK)
    ax.set_yticks(np.log10([0.05, 0.1, 0.3, 1, 3]), ["0.05", "0.1", "0.3", "1", "3"])
    ax.set_ylim(np.log10(0.04), np.log10(4))
    ax.set_xticks(range(K), labels, rotation=90, ha="center")
    state_ticklabels(ax, names, "x")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("bout duration (s)")
    ax.set_title("bout durations (all sessions; bar and value: median)", loc="left")
    panel(ax, "d", fx=0.01)

    # (e) occupancy per condition
    ax = fig.add_subplot(r2[1])
    conds = A.conditions
    occ = np.array([[np.mean([np.mean(s["z"] == k) for s in sessions if s["condition"] == c])
                     for c in conds] for k in range(K)])
    heat(ax, occ * 100, labels, ["Control", "Clozapine", "Amph.", "Amph. +\nClozapine"],
         SEQUENTIAL, 0, 50, lambda v: f"{v:.0f}" if v >= 1 else "<1", 0)
    state_ticklabels(ax, names)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_title("occupancy (% of time,\nmean over sessions)", loc="left")
    panel(ax, "e", fx=0.6)

    # (f) example: speed + state sequence
    s = next(x for x in sessions if x["session"] == example)
    t0, _ = example_window(s, states)
    t1 = t0 + int(60 * A.fs)
    sub = gs[2].subgridspec(2, 1, height_ratios=[1, 0.22], hspace=0.08)
    t = np.arange(t1 - t0) / A.fs
    ax = fig.add_subplot(sub[0])
    ax.plot(t, s["feat"].speed.values[t0:t1], color=INK2, lw=0.7)
    ax.set_xlim(0, 60)
    ax.set_ylabel("speed (cm/s)")
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("example: torso speed and state sequence (same session as fig. 1e, first "
                 "60 s of its window)", loc="left")
    panel(ax, "f", fx=0.01, y=1.08)
    ax2 = fig.add_subplot(sub[1], sharex=ax)
    cmap = ListedColormap([STATE_COLORS[n] for n in names])
    ax2.imshow(s["z"][t0:t1][None], cmap=cmap, vmin=-0.5, vmax=K - 0.5, aspect="auto",
               extent=(0, 60, 0, 1), interpolation="nearest")
    ax2.set_yticks([])
    ax2.set_xlabel("time (s)")
    for sp in ("left",):
        ax2.spines[sp].set_visible(False)
    present = sorted(set(s["z"][t0:t1]))
    ax2.legend(handles=[Patch(color=STATE_COLORS[names[k]], label=STATE_LABELS[names[k]])
                        for k in present], ncol=len(present), loc="upper left",
               bbox_to_anchor=(0, -2.3), frameon=False, handlelength=1, columnspacing=1.2)
    fig.text(0.15, 0.006, "* rearing / tracking artefact, excluded from the analyses",
             ha="left", va="bottom", fontsize=6, color=MUTED)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.1)
    save(fig, "fig_pose_hmm")

# ---------------------------------------------------------------------------
# figure 3: typical bout per state
# ---------------------------------------------------------------------------

def typical_bout(b, k):
    """Bout of state k whose mean feature vector is closest to the median over bouts
    (duration between the median and the 90th percentile, at least 0.3 s, no
    interpolated keypoints)."""
    d = b[b.state == k]
    lo, hi = max(d.dur.median(), 0.3), max(d.dur.quantile(0.9), 0.35)
    c = d[d.dur.between(lo, hi) & (d.interp == 0)]
    if not len(c):
        c = d[d.dur >= lo].nsmallest(50, "interp")
    m = d[A.features].median().values
    return c.iloc[np.argmin(((c[A.features].values - m) ** 2).sum(1))]

def egocentric(seq, ix):
    """Keypoints (T, 7, 2, cm, image coordinates) -> coordinates with y up, origin at the
    start torso and the start body axis pointing up."""
    q = seq * np.array([1, -1])
    torso = q[0, [ix["neck"], ix["hip1"], ix["hip2"], ix["tail_base"]]].mean(0)
    ax = q[0, ix["neck"]] - q[0, ix["tail_base"]]
    a = np.pi / 2 - np.arctan2(ax[1], ax[0])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (q - torso) @ R.T

def fig_states(states, sessions, b):
    K = len(states)
    names = states.name.tolist()
    order = [k for fam in (("rest", "quiet", "shift"), ("slow_walk", "walk", "run", "run_fast"),
                           ("reorient", "turn_fast", "rear_artifact"))
             for k in range(K) if names[k] in fam]
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 4.6))
    cache = {}
    for ax, k in zip(axes.flat, order):
        bt = typical_bout(b, k)
        sess = sessions[int(bt.sess)]["session"]
        if sess not in cache:
            cache[sess] = keypoints(sess)
        kp, _, _, _, ix = cache[sess]
        seq = egocentric(kp[int(bt.start):int(bt.end) + 1], ix)   # incl. the first frame after
        col = STATE_COLORS[names[k]]
        # poses over the bout, coloured from light grey (start) to the state colour (end)
        idx = np.unique(np.linspace(0, len(seq) - 1, min(len(seq), 6)).round().astype(int))
        c0, c1 = np.array(to_rgb(START_GREY)), np.array(to_rgb(col))
        for i, t in enumerate(idx):
            w = i / (len(idx) - 1)
            last = i == len(idx) - 1
            skeleton(ax, seq[t], ix, color=(1 - w) * c0 + w * c1, lw=1.3 if last else 0.9,
                     ms=12 if last else 5, edge="white", z=3 + i)
        for kpn in ("nose", "tail_base"):
            ax.plot(seq[:, ix[kpn], 0], seq[:, ix[kpn], 1], color=INK2, lw=0.5, ls=(0, (1.5, 1)),
                    zorder=4)
        tors = seq[:, [ix["neck"], ix["hip1"], ix["hip2"], ix["tail_base"]]].mean(1)
        f = sessions[int(bt.sess)]["feat"].iloc[int(bt.start):int(bt.end)]
        disp = np.linalg.norm(tors[-1] - tors[0])
        nose = np.linalg.norm(np.diff(seq[:, ix["nose"]], axis=0), axis=1).sum()
        net = f.turn.sum() / A.fs
        ax.set_title(STATE_LABELS[names[k]] + (" (excl.)" if states.excluded[k] else ""),
                     color=col, fontweight="bold",
                     loc="left")
        ax.text(0.02, 0.0, f"{bt.dur:.2f} s · torso {disp:.1f} cm\nnose {nose:.1f} cm · "
                f"turn {net:+.0f}°".replace("-", "−"),
                transform=ax.transAxes, fontsize=5.8, color=INK2, va="bottom")
        ax.set_aspect("equal")
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-6.5, 9.5)
        ax.axis("off")
    ax = axes.flat[0]
    ax.plot([-5, -4], [8.8, 8.8], color=INK, lw=1)
    ax.text(-4.5, 9.0, "1 cm", ha="center", va="bottom", fontsize=5.8, color=INK)
    fig.legend(handles=[plt.Line2D([], [], color=START_GREY, marker="o", lw=0.9, ms=3,
                                   mec="white", label="bout start (light grey)"),
                        plt.Line2D([], [], color=INK2, marker="o", lw=1.3, ms=4.5, mec="white",
                                   label="→ bout end (state colour), poses at equal time steps"),
                        plt.Line2D([], [], color=INK2, lw=0.5, ls=(0, (1.5, 1)),
                                   label="nose / tail-base path")],
               loc="lower center", ncol=3, frameon=False, handlelength=2)
    fig.suptitle("Typical bout per state (egocentric: start body axis pointing up; one bout "
                 "each, not averaged)", x=0.02, ha="left", fontsize=8, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08, wspace=0.05, hspace=0.25)
    save(fig, "fig_pose_states")


if __name__ == "__main__":
    states, sessions = load_data()
    b = bouts(sessions)
    fig_features(states, sessions)
    fig_hmm(states, sessions, b)
    fig_states(states, sessions, b)
