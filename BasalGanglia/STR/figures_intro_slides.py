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


if __name__ == "__main__":
    intro_circuit()
    intro_disease()
    intro_problem()
