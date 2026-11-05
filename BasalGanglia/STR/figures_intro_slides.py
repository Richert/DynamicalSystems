"""Schematics of the intro slides (striatal microcircuit, disease states, the drug-development
gap), in the slide style of the other figure scripts (white background, Nimbus Sans).

  intro_circuit   cortico-basal ganglia loop with the direct / indirect pathway model and the
                  striatal microcircuit (D1 / D2 SPNs, cholinergic and fast-spiking interneurons,
                  dopamine from SNc)
  intro_disease   activity changes hypothesized in Parkinson's disease vs schizophrenia
  intro_problem   drug -> receptors -> striatal circuit -> function / behaviour, and the
                  behaviour-only shortcut of drug development

  conda run -n parker python figures_intro_slides.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

out_dir = os.environ.get("SLIDE_DIR", "/home/rgast/OneDrive/presentations/parker/images")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
D1C, D2C = "#1f8a70", "#8a4fb5"                      # D1 / D2 SPNs
DAC, ACHC, FSIC = "#d1860a", "#c2185b", "#5f6b7a"     # dopamine, acetylcholine, FSI
EXC, INH = "#3a3936", "#c4372f"                       # glutamatergic / GABAergic projections
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 8, "pdf.fonttype": 42,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

def save(fig, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.pdf")
    fig.savefig(f"{out_dir}/{name}.png", dpi=200)
    plt.close(fig)
    print(f"saved {out_dir}/{name}.pdf/.png")

def node(ax, xy, text, color=INK2, w=0.15, h=0.075, fill=None, fs=7.5, bold=False, alpha=1.0,
         tc=None):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=fill if fill is not None else "white", ec=color, lw=1.1,
                                alpha=alpha, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc or color,
            fontweight="bold" if bold else "normal", zorder=4)

def arrow(ax, a, b, kind="exc", lw=1.2, color=None, rad=0.0, shrink=14, ls="-", z=2):
    """kind exc: triangle head (glutamate); inh: bar head (GABA); mod: open head, dashed."""
    color = color or (EXC if kind == "exc" else INH if kind == "inh" else DAC)
    style = {"exc": "-|>,head_length=4,head_width=2.4", "inh": "-[,widthB=3.0,lengthB=0",
             "mod": "-|>,head_length=3.5,head_width=2.2"}[kind]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink,
                                 linestyle=ls, zorder=z, mutation_scale=1.6 if kind != "inh" else 1.4))

def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

def legend_items(ax, x, y, dy=0.06):
    items = [("exc", "excitatory (glutamate)", EXC), ("inh", "inhibitory (GABA)", INH),
             ("mod", "dopamine", DAC), ("mod", "acetylcholine", ACHC)]
    for i, (k, t, c) in enumerate(items):
        yy = y - i * dy
        arrow(ax, (x, yy), (x + 0.06, yy), kind=k, color=c, shrink=0, lw=1.1,
              ls="--" if k == "mod" else "-")
        ax.text(x + 0.09, yy, t, va="center", fontsize=6.5, color=INK2)

# ---------------------------------------------------------------------------

def intro_circuit():
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    fig.subplots_adjust(0, 0, 1, 1)
    blank(ax)
    P = dict(ctx=(0.42, 0.93), d1=(0.24, 0.66), d2=(0.60, 0.66), chi=(0.42, 0.735), fsi=(0.42, 0.585),
             snc=(0.9, 0.66), gpe=(0.60, 0.43), stn=(0.60, 0.23), gpi=(0.24, 0.33), thal=(0.24, 0.08))
    # striatum
    ax.add_patch(FancyBboxPatch((0.11, 0.535), 0.62, 0.25, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc="#f6f5f0", ec=GRID, lw=1.0, zorder=0))
    ax.text(0.12, 0.765, "striatum", fontsize=7, color=MUTED, va="center")
    node(ax, P["ctx"], "cortex", INK2, w=0.18)
    node(ax, P["d1"], "D1 SPN", D1C, w=0.15, bold=True)
    node(ax, P["d2"], "D2 SPN", D2C, w=0.15, bold=True)
    node(ax, P["chi"], "ChI", ACHC, w=0.08, h=0.05, fs=6.5)
    node(ax, P["fsi"], "FSI", FSIC, w=0.08, h=0.05, fs=6.5)
    node(ax, P["snc"], "SNc", DAC, w=0.12, bold=True)
    node(ax, P["gpe"], "GPe", INK2, w=0.12)
    node(ax, P["stn"], "STN", INK2, w=0.12)
    node(ax, P["gpi"], "GPi / SNr", INK2, w=0.16)
    node(ax, P["thal"], "thalamus", INK2, w=0.16)
    # projections
    arrow(ax, P["ctx"], P["d1"], "exc", rad=0.1)
    arrow(ax, P["ctx"], P["d2"], "exc", rad=-0.1)
    arrow(ax, P["d1"], P["gpi"], "inh", lw=1.6, color=INH)
    arrow(ax, P["d2"], P["gpe"], "inh", lw=1.6)
    arrow(ax, P["gpe"], P["stn"], "inh")
    arrow(ax, P["stn"], P["gpi"], "exc", shrink=30)
    arrow(ax, P["gpi"], P["thal"], "inh")
    ax.plot([0.16, 0.035, 0.035], [0.08, 0.08, 0.93], color=EXC, lw=1.2, zorder=2,
            solid_capstyle="round")                     # thalamus -> cortex (outside the loop)
    arrow(ax, (0.035, 0.93), (0.33, 0.93), "exc", shrink=0)
    # dopamine
    arrow(ax, P["snc"], P["d2"], "mod", color=DAC, ls="--", lw=1.2)
    arrow(ax, P["snc"], P["d1"], "mod", color=DAC, ls="--", lw=1.2, rad=0.32)
    ax.text(0.76, 0.69, "D2 (−)", fontsize=6.5, color=DAC, ha="center")
    ax.text(0.62, 0.825, "D1 (+)", fontsize=6.5, color=DAC, ha="center")
    # interneurons
    arrow(ax, P["chi"], P["d1"], "mod", color=ACHC, ls="--", lw=0.9, shrink=8)
    arrow(ax, P["chi"], P["d2"], "mod", color=ACHC, ls="--", lw=0.9, shrink=8)
    arrow(ax, P["fsi"], P["d1"], "inh", lw=0.9, shrink=8)
    arrow(ax, P["fsi"], P["d2"], "inh", lw=0.9, shrink=8)
    # pathway labels
    ax.text(0.115, 0.47, "direct\npathway\n\"go\"", fontsize=7, color=D1C, ha="center", va="center",
            fontweight="bold")
    ax.text(0.77, 0.33, "indirect\npathway\n\"no-go\"", fontsize=7, color=D2C, ha="center", va="center",
            fontweight="bold")
    legend_items(ax, 0.7, 0.2)
    save(fig, "intro_circuit")

# ---------------------------------------------------------------------------

def mini_circuit(ax, title, da, d1, d2, out, thal, da_text, effect, color):
    """Simplified loop; activity levels in [-1, 1] (0: normal) set fill and line width."""
    blank(ax)
    shade = lambda c, a: tuple(1 - (1 - np.array(matplotlib.colors.to_rgb(c))) * (0.18 + 0.32 * (a + 1)))
    lw = lambda a: 1.3 * (1 + 0.9 * a)
    P = dict(da=(0.82, 0.8), d1=(0.28, 0.62), d2=(0.66, 0.62), ind=(0.66, 0.38), gpi=(0.28, 0.38),
             thal=(0.28, 0.14))
    node(ax, P["da"], "dopamine", DAC, w=0.24, fill=shade(DAC, da), bold=True)
    node(ax, P["d1"], "D1 SPN", D1C, w=0.2, fill=shade(D1C, d1), bold=True)
    node(ax, P["d2"], "D2 SPN", D2C, w=0.2, fill=shade(D2C, d2), bold=True)
    node(ax, P["ind"], "GPe → STN", INK2, w=0.22, fs=6.8)
    node(ax, P["gpi"], "GPi / SNr", INK2, w=0.22, fill=shade("#7a7974", out), fs=7)
    node(ax, P["thal"], "thalamus → cortex", INK2, w=0.34, fill=shade("#7a7974", thal), fs=7)
    arrow(ax, P["da"], P["d1"], "mod", color=DAC, ls="--", lw=lw(da), rad=0.25)
    arrow(ax, P["da"], P["d2"], "mod", color=DAC, ls="--", lw=lw(da))
    arrow(ax, P["d1"], P["gpi"], "inh", lw=lw(d1))
    arrow(ax, P["d2"], P["ind"], "inh", lw=lw(d2))
    arrow(ax, P["ind"], P["gpi"], "exc", lw=lw(d2))
    arrow(ax, P["gpi"], P["thal"], "inh", lw=lw(out))
    for key, a in (("da", da), ("d1", d1), ("d2", d2), ("gpi", out), ("thal", thal)):
        if a:
            x, y = P[key]
            w = {"da": 0.24, "thal": 0.34, "gpi": 0.22}.get(key, 0.2)
            ax.text(x + w / 2 + 0.015, y + (0.04 if key == "gpi" else 0), "↑" if a > 0 else "↓",
                    fontsize=12, va="center",
                    color=INK, fontweight="bold")
    ax.text(0.0, 1.0, title, fontsize=10, fontweight="bold", color=color, va="top")
    ax.text(0.98, 0.93, da_text, fontsize=6.5, color=INK2, ha="right", va="top")
    ax.text(0.62, 0.17, effect, fontsize=7, color=color, va="center", ha="left", linespacing=1.3)

def intro_disease():
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.55))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03, wspace=0.08)
    mini_circuit(axes[0], "Parkinson's disease", da=-1, d1=-0.8, d2=0.8, out=0.8, thal=-0.8,
                 da_text="loss of SNc\ndopamine neurons", color="#2a5d9f",
                 effect="hypokinesia:\nbradykinesia,\nrigidity, tremor")
    mini_circuit(axes[1], "Schizophrenia", da=1, d1=0.7, d2=-0.7, out=0, thal=0,
                 da_text="excess striatal dopamine\n(synthesis, release)", color="#b5382f",
                 effect="positive symptoms:\npsychosis, delusions,\nhallucinations")
    save(fig, "intro_disease")

# ---------------------------------------------------------------------------

def intro_problem():
    fig, ax = plt.subplots(figsize=(6.0, 2.45))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.42)
    ax.axis("off")
    y = 0.27
    # drug
    node(ax, (0.07, y), "drug", INK, w=0.1, h=0.07, bold=True, fs=8.5)
    # receptor profile
    rx = 0.27
    ax.text(rx, y + 0.115, "receptor affinities", ha="center", fontsize=7, color=INK2)
    recs = [("D1", D1C), ("D2", D2C), ("5-HT$_{2A}$", "#2a78d6"), ("M1/M4", ACHC), ("H1", MUTED), ("α1", MUTED)]
    for i, (t, c) in enumerate(recs):
        xx = rx - 0.075 + (i % 3) * 0.075
        yy = y + 0.04 - (i // 3) * 0.075
        node(ax, (xx, yy), t, c, w=0.065, h=0.05, fs=6.5)
    # striatal circuit
    cx = 0.58
    ax.add_patch(FancyBboxPatch((cx - 0.13, y - 0.085), 0.26, 0.17, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc="#f6f5f0", ec=GRID, lw=1.0, zorder=0))
    ax.text(cx, y + 0.115, "striatal microcircuit", ha="center", fontsize=7, color=INK2)
    node(ax, (cx - 0.065, y + 0.035), "D1 SPN", D1C, w=0.1, h=0.05, fs=6.5, bold=True)
    node(ax, (cx + 0.065, y + 0.035), "D2 SPN", D2C, w=0.1, h=0.05, fs=6.5, bold=True)
    for i, (t, c) in enumerate((("DA", DAC), ("ACh", ACHC), ("5-HT", "#2a78d6"))):
        node(ax, (cx - 0.075 + i * 0.075, y - 0.04), t, c, w=0.06, h=0.045, fs=6.3)
    # function
    node(ax, (0.9, y), "function /\nbehaviour", INK, w=0.15, h=0.1, bold=True, fs=8)
    # arrows with the two open questions
    for (a, b, q) in (((0.12, y), (0.17, y), None), ((0.37, y), (0.44, y), "1"),
                      ((0.72, y), (0.82, y), "2")):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_length=5,head_width=3", color=INK2,
                                     lw=1.4, mutation_scale=1.6, shrinkA=2, shrinkB=2))
        if q:
            xm = (a[0] + b[0]) / 2
            ax.text(xm, y + 0.03, "?", ha="center", fontsize=15, color="#c4372f", fontweight="bold")
            ax.add_patch(Circle((xm, y - 0.05), 0.017, fc="#c4372f", ec="none"))
            ax.text(xm, y - 0.05, q, ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")
    # behaviour-only shortcut
    ax.add_patch(FancyArrowPatch((0.07, y - 0.045), (0.9, y - 0.06), arrowstyle="-|>,head_length=5,head_width=3",
                                 connectionstyle="arc3,rad=0.28", color="#2a78d6", lw=1.4, ls="--",
                                 mutation_scale=1.6, shrinkA=6, shrinkB=10))
    ax.text(0.485, 0.045, "drug development: behavioural screens (e.g. amphetamine-induced hyperlocomotion)",
            ha="center", fontsize=7, color="#2a78d6")
    save(fig, "intro_problem")

def intro_mediation():
    """Direct drug effect on SPN activity vs an effect mediated by behaviour (2 build steps)."""
    fig, ax = plt.subplots(figsize=(5.0, 2.55))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.51)
    ax.axis("off")
    P = dict(drug=(0.1, 0.12), spn=(0.88, 0.12), beh=(0.49, 0.37))
    node(ax, P["drug"], "drug", INK, w=0.14, h=0.08, bold=True, fs=9)
    node(ax, P["spn"], "SPN activity", D1C, w=0.2, h=0.08, bold=True, fs=9)
    node(ax, P["beh"], "behaviour", "#d1860a", w=0.18, h=0.08, bold=True, fs=9)
    # arrow ends placed explicitly with a gap to every box edge (boxes: centre +- w/2, h/2)
    direct = FancyArrowPatch((0.195, 0.12), (0.755, 0.12), arrowstyle="-|>,head_length=6,head_width=3.5",
                             color=INK, lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=1.6)
    ax.add_patch(direct)
    t_dir = ax.text(0.49, 0.075, "direct effect on the striatal circuit?", ha="center", fontsize=8,
                    color=INK)
    m1 = FancyArrowPatch((0.15, 0.185), (0.37, 0.32), arrowstyle="-|>,head_length=6,head_width=3.5",
                         color="#d1860a", lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=1.6)
    m2 = FancyArrowPatch((0.61, 0.32), (0.815, 0.185), arrowstyle="<|-|>,head_length=6,head_width=3.5",
                         color="#d1860a", lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=1.6)
    ax.add_patch(m1)
    ax.add_patch(m2)
    t_med = ax.text(0.49, 0.5, "or mediated by the change in behaviour?", ha="center", fontsize=8,
                    color="#d1860a", va="top")
    steps = [{direct, t_dir}, {direct, t_dir, m1, m2, t_med}]
    managed = set().union(*steps)
    for k, vis in enumerate(steps, 1):
        for a in managed:
            a.set_visible(a in vis)
        fig.savefig(f"{out_dir}/intro_mediation_s{k}.pdf")
        fig.savefig(f"{out_dir}/intro_mediation_s{k}.png", dpi=150)
    plt.close(fig)
    print(f"saved {out_dir}/intro_mediation_s1..2")

def draw_mouse(ax, x, y, s=1.0, mode="walk", color="#6b6a66"):
    """Cartoon mouse (top-ish side view) at (x, y), scale s; mode walk / run / rest."""
    from matplotlib.patches import Ellipse, Circle, Arc
    if mode == "rest":                                       # curled up, eyes closed, zzz
        ax.add_patch(Ellipse((x, y), 0.16 * s, 0.1 * s, fc=color, ec="none", zorder=3))
        ax.add_patch(Circle((x + 0.065 * s, y + 0.01 * s), 0.035 * s, fc=color, ec="none", zorder=3))
        ax.add_patch(Circle((x + 0.05 * s, y + 0.05 * s), 0.016 * s, fc="#e8b4b8", ec=color, lw=0.5, zorder=4))
        ax.add_patch(Arc((x - 0.02 * s, y - 0.02 * s), 0.2 * s, 0.12 * s, theta1=180, theta2=330, color=color,
                         lw=1.0, zorder=2))
        for i, z in enumerate("zzZ"):
            ax.text(x + (0.11 + 0.035 * i) * s, y + (0.0 + 0.025 * i) * s, z, fontsize=5 + i, color=INK2,
                    fontweight="bold")
        return
    ax.add_patch(Ellipse((x, y), 0.2 * s, 0.09 * s, fc=color, ec="none", zorder=3))
    ax.add_patch(Ellipse((x + 0.11 * s, y + 0.01 * s), 0.09 * s, 0.06 * s, angle=-10, fc=color, ec="none", zorder=3))
    ax.add_patch(Circle((x + 0.09 * s, y + 0.045 * s), 0.022 * s, fc="#e8b4b8", ec=color, lw=0.5, zorder=4))
    ax.add_patch(Circle((x + 0.135 * s, y + 0.02 * s), 0.007 * s, fc="white", ec="none", zorder=5))
    tx = np.linspace(0, 1, 30)
    ax.plot(x - 0.1 * s - 0.12 * s * tx, y - 0.01 * s + 0.03 * s * np.sin(tx * 5), color=color, lw=1.0, zorder=2)
    leg_dx = (0.035, -0.04) if mode == "walk" else (0.06, -0.07)
    for lx in (-0.055, 0.045):
        for d in leg_dx:
            ax.plot([x + lx * s, x + (lx + d * 0.5) * s], [y - 0.035 * s, y - 0.07 * s], color=color, lw=0.9,
                    zorder=2)
    if mode == "run":                                        # speed lines
        for i in range(3):
            ax.plot([x - (0.26 + 0.02 * i) * s, x - (0.16 + 0.02 * i) * s], [y + (0.03 - 0.03 * i) * s] * 2,
                    color=INK2, lw=0.8)

def summary_findings():
    """Cartoon summary: per condition the behaviour, the share of speed-tuned SPNs (D1, D2) with a
    mini tuning curve, the session-wide SPN rate and the SPN rate while moving (matched speed), both
    relative to Control."""
    rows = [("Control", "#2a78d6", "walk", (0.56, 0.61), (1.0, 1.0), (1.0, 1.0), "tuned"),
            ("Amph.", "#eda100", "run", (0.09, 0.03), (1.35, 0.88), (0.66, 0.48), "flat"),
            ("Amph. +\nclozapine", "#e34948", "walk", (0.22, 0.04), (0.81, 0.65), (0.75, 0.49), "D1"),
            ("Amph. +\nhaloperidol", "#b5382f", "rest", (0.37, 0.24), (0.44, 0.44), (0.92, 0.81), "partial")]
    fig, ax = plt.subplots(figsize=(6.15, 2.85))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.465)
    ax.axis("off")
    bar_x, bar_scale = {"session": 0.625, "moving": 0.815}, 0.1          # width per 100 %
    cols = [(0.2, "behaviour"), (0.445, "speed-tuned SPNs"), (bar_x["session"] + 0.07, "session rate"),
            (bar_x["moving"] + 0.075, "rate while moving")]
    for x, t in cols:
        ax.text(x, 0.445, t, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
    ys = [0.36, 0.265, 0.17, 0.075]
    for (name, col, mode, tuned, sess, mov, curve), y in zip(rows, ys):
        ax.add_patch(FancyBboxPatch((0.005, y - 0.04), 0.99, 0.08, boxstyle="round,pad=0,rounding_size=0.012",
                                    fc="#f6f5f0", ec="none", zorder=0))
        ax.text(0.015, y, name, ha="left", va="center", fontsize=7.2, fontweight="bold", color=col,
                linespacing=1.05)
        draw_mouse(ax, 0.205, y + 0.005, s=0.4, mode=mode)
        for g, (spn, c_, sh) in enumerate((("D1", D1C, tuned[0]), ("D2", D2C, tuned[1]))):
            yy = y + 0.017 - 0.034 * g
            ax.text(0.33, yy, spn, ha="right", va="center", fontsize=6.3, color=c_, fontweight="bold")
            nf = int(round(sh * 10))
            for i in range(10):
                ax.add_patch(Circle((0.343 + i * 0.0165, yy), 0.0063, fc=c_ if i < nf else "white", ec=c_, lw=0.6,
                                    zorder=3))
        x0, w, h = 0.52, 0.05, 0.05
        sx = np.linspace(0, 1, 30)
        shape = {"tuned": 0.15 + 0.85 * sx ** 1.5, "flat": 0.45 + 0.08 * sx,
                 "D1": 0.3 + 0.45 * sx ** 1.5, "partial": 0.2 + 0.65 * sx ** 1.5}[curve]
        ax.plot([x0, x0, x0 + w], [y + h / 2, y - h / 2, y - h / 2], color=INK2, lw=0.6)
        ax.plot(x0 + w * sx, y - h / 2 + h * 0.9 * shape, color=col, lw=1.4)
        if y == ys[0]:
            ax.text(x0 + w / 2, y - h / 2 - 0.012, "speed", ha="center", va="top", fontsize=5.3, color=INK2)
            ax.text(x0 - 0.005, y, "rate", ha="right", va="center", fontsize=5.3, color=INK2, rotation=90)
        for key, vals in (("session", sess), ("moving", mov)):
            bx = bar_x[key]
            for g, (c_, v) in enumerate(((D1C, vals[0]), (D2C, vals[1]))):
                yy = y + 0.015 - 0.03 * g
                ax.add_patch(FancyBboxPatch((bx, yy - 0.009), bar_scale * v, 0.018,
                                            boxstyle="round,pad=0,rounding_size=0.004", fc=c_, ec="none", zorder=3))
                if name != "Control":
                    ax.text(bx + bar_scale * v + 0.005, yy, f"{v:.0%}", va="center", fontsize=5.8, color=INK2,
                            zorder=5, bbox=dict(fc="#f6f5f0", ec="none", pad=0.4))
            ax.plot([bx + bar_scale] * 2, [y - 0.035, y + 0.035], color=MUTED, lw=0.6, ls=(0, (2, 1.5)), zorder=4)
    for key in ("session", "moving"):
        ax.text(bar_x[key] + bar_scale, ys[-1] - 0.05, "Control", ha="center", fontsize=5.3, color=MUTED)
    fig.savefig(f"{out_dir}/summary_findings.pdf")
    fig.savefig(f"{out_dir}/summary_findings.png", dpi=200)
    plt.close(fig)
    print(f"saved {out_dir}/summary_findings.pdf/.png")

def summary_outlook():
    """Second summary / outlook: SD of SPN rates per condition, the dopamine rate-distribution
    hypothesis (cartoon), evidence from our data, next steps."""
    from scipy.stats import norm as gauss
    fig = plt.figure(figsize=(6.15, 2.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.465)
    ax.axis("off")
    for x, t in ((0.15, "rate spread"), (0.5, "hypothesis"), (0.85, "our data")):
        ax.text(x, 0.445, t, ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
    # column 1: narrow vs wide distribution + SD of log rates relative to Control
    a1 = fig.add_axes([0.04, 0.5, 0.25, 0.33])
    xr = np.linspace(-3, 3, 200)
    a1.fill_between(xr, gauss.pdf(xr, 0, 0.6), color="#2a78d6", alpha=0.25, lw=0)
    a1.plot(xr, gauss.pdf(xr, 0, 0.6), color="#2a78d6", lw=1.2)
    a1.plot(xr, 0.6 / 0.95 * gauss.pdf(xr, 0, 0.95), color="#e34948", lw=1.2)
    a1.text(0.0, 0.72, "Control", ha="center", fontsize=6, color="#2a78d6")
    a1.text(1.5, 0.22, "drugs", ha="left", fontsize=6, color="#e34948")
    a1.set_xlabel("log rate", fontsize=6, labelpad=1)
    a1.set_xticks([])
    a1.set_yticks([])
    for sp in ("top", "right", "left"):
        a1.spines[sp].set_visible(False)
    a2 = fig.add_axes([0.075, 0.2, 0.215, 0.21])
    cond = ["Amph.", "A + Cloz.", "A + Halo."]
    d1, d2 = [1.09, 1.29, 1.40], [1.19, 1.40, 1.40]
    xx = np.arange(3)
    a2.bar(xx - 0.18, np.array(d1) - 1, 0.34, bottom=1, color=D1C, label="D1")
    a2.bar(xx + 0.18, np.array(d2) - 1, 0.34, bottom=1, color=D2C, label="D2")
    a2.axhline(1, color=MUTED, lw=0.6, ls=(0, (2, 1.5)))
    a2.set_ylim(0.95, 1.5)
    a2.set_yticks([1, 1.2, 1.4], ["Ctrl", "+20%", "+40%"])
    a2.set_xticks(xx, cond, fontsize=5.6, rotation=25, ha="right")
    a2.tick_params(axis="y", labelsize=5.6, length=2)
    a2.tick_params(axis="x", length=0)
    a2.set_title("SD of log rate", fontsize=6.3, loc="left", pad=2)
    a2.legend(frameon=False, fontsize=5.5, loc="upper left", handlelength=0.8, borderaxespad=0)
    for sp in ("top", "right"):
        a2.spines[sp].set_visible(False)
    # column 2: hypothesis cartoons
    for k, (title, m1, m2, sd) in enumerate((("low dopamine / rest", -0.9, 0.9, 0.42),
                                              ("high dopamine", 0.35, -0.35, 0.85))):
        ah = fig.add_axes([0.365, 0.55 - 0.43 * k, 0.27, 0.28])
        p1, p2 = gauss.pdf(xr, m1, sd), gauss.pdf(xr, m2, sd)
        ah.fill_between(xr, (p1 + p2) / 2, color="#cfcdc4", alpha=0.7, lw=0)        # combined
        ah.plot(xr, p1 / 2, color=D1C, lw=1.4)
        ah.plot(xr, p2 / 2, color=D2C, lw=1.4)
        ah.set_xticks([])
        ah.set_yticks([])
        ah.set_ylim(0, 0.55)
        for sp in ("top", "right", "left"):
            ah.spines[sp].set_visible(False)
        ah.set_title(title, fontsize=6.8, loc="left", pad=1, color=INK)
        ah.set_xlabel("log rate", fontsize=5.8, labelpad=1)
        if k == 0:
            ah.text(m1, 0.5, "D1", color=D1C, fontsize=6.5, ha="center", fontweight="bold")
            ah.text(m2, 0.5, "D2", color=D2C, fontsize=6.5, ha="center", fontweight="bold")
            ah.text(3.0, 0.3, "combined:\nbimodal", fontsize=5.6, color=INK2, ha="right")
        else:
            ah.annotate("", xy=(m1 + 0.7, 0.3), xytext=(m1 - 0.1, 0.3),
                        arrowprops=dict(arrowstyle="-|>", color=D1C, lw=1.0, mutation_scale=6))
            ah.annotate("", xy=(m2 - 0.7, 0.36), xytext=(m2 + 0.1, 0.36),
                        arrowprops=dict(arrowstyle="-|>", color=D2C, lw=1.0, mutation_scale=6))
            ah.text(3.0, 0.43, "combined:\nbroad, unimodal", fontsize=5.6, color=INK2, ha="right")
    ax.annotate("", xy=(0.56, 0.222), xytext=(0.56, 0.252), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0,
                                                                         mutation_scale=7))
    # column 3: evidence
    def check(x, y, good):
        if good:
            ax.plot([x - 0.011, x - 0.003, x + 0.012], [y, y - 0.012, y + 0.014], color="#1f8a70", lw=2.2,
                    solid_capstyle="round", solid_joinstyle="round")
        else:
            for sgn in (1, -1):
                ax.plot([x - 0.01, x + 0.01], [y - 0.012 * sgn, y + 0.012 * sgn], color="#c4372f", lw=2.2,
                        solid_capstyle="round")
    items = [(True, "both distributions widen\n(all drug conditions)"), (True, "Amph.: D1 ↑, D2 ↓\n(session-wide)"),
             (False, "D2 > D1 at baseline"), (False, "bimodal at baseline")]
    for i, (good, t) in enumerate(items):
        y = 0.385 - i * 0.07
        check(0.71, y, good)
        ax.text(0.735, y, t, fontsize=6.3, va="center", color=INK, linespacing=1.1)
    ax.text(0.7, 0.095, "D1 / D2 SPNs recorded in different mice;\nAmph. D1 ↑ is behaviour-mediated", fontsize=5.6,
            color=MUTED, va="center", linespacing=1.15)
    # bottom row: next steps
    ax.add_patch(FancyBboxPatch((0.005, 0.005), 0.99, 0.04, boxstyle="round,pad=0,rounding_size=0.008",
                                fc="#f6f5f0", ec="none", zorder=0))
    steps = ["next: other dopamine drugs in the data set", "network model: tuned SPNs + DA-set excitability",
             "context-specific striatal function"]
    xs = [0.02, 0.385, 0.765]
    for i, (x, t) in enumerate(zip(xs, steps)):
        ax.text(x, 0.025, t, fontsize=6.3, va="center", color="#2a5d9f" if i == 0 else INK,
                fontweight="bold" if i == 0 else "normal")
        if i < 2:
            ax.annotate("", xy=(xs[i + 1] - 0.01, 0.025), xytext=(xs[i + 1] - 0.035, 0.025),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=6))
    fig.savefig(f"{out_dir}/summary_outlook.pdf")
    fig.savefig(f"{out_dir}/summary_outlook.png", dpi=200)
    plt.close(fig)
    print(f"saved {out_dir}/summary_outlook.pdf/.png")


if __name__ == "__main__":
    intro_circuit()
    intro_disease()
    intro_problem()
    intro_mediation()
    summary_findings()
    summary_outlook()
