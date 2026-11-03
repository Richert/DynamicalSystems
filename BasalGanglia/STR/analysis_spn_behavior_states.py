"""SPN activity conditioned on the behaviour labels (pose AR-HMM, analysis_pose_arhmm.py) from
the SPN data file (build_spn_dataset.py).

Per recording of the analysis conditions, per cell and behaviour label (label_clean: bouts >=
min_bout_s, artefacts excluded), over the label's frames:
  rate        inferred events / s                                   (spike-based)
  trace       mean denoised trace C / the cell's noise level         (trace-based)
  s_rate      mean deconvolved activity S / noise x fs (rate proxy)  (spike-based, threshold-free)
with skip_s > 0 the first skip_s of every bout are left out (calcium carry-over from the
previous behaviour). A (recording, label) pair is used only with >= min_label_s in the label and
>= min_bouts bouts. Per recording and label also the population values: median over cells and the
share of active cells (>= 1 event).

Outputs (OUT_DIR): cell_label_table.csv(.gz), recording_label_table.csv, summary figures.

  conda run -n parker python analysis_spn_behavior_states.py [skip_s]
"""
import os
import sys
import numpy as np
import pandas as pd
import h5py

dataset = os.environ.get("DATASET", "/home/rgast/data/parker_data/dataset/clozapine_spn.h5")
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/spn_behavior_states")
conditions = ["Control", "Amphetamine", "A + Clozapine low", "A + Clozapine high"]
fs = 20.0
min_bout_s = 0.2                 # (label_clean already drops shorter bouts)
min_label_s = 20.0
min_bouts = 10
skip_s = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

def bouts_of(lab):
    """Start / end / label of each run of equal labels (-1 runs included)."""
    st = np.flatnonzero(np.diff(np.r_[-2, lab]) != 0)
    en = np.r_[st[1:], len(lab)]
    return st, en, lab[st]

def recording_tables(g, day_attrs, rec_name):
    r = g[rec_name]
    cond = r.attrs["condition"]
    beh = r["behaviour"]
    if cond not in conditions or not beh.attrs.get("available", False):
        return [], []
    names = list(beh.attrs["label_names"])
    lab = beh["label_clean"][:]
    C = r["C"][:]
    sn = r["sn"][:]
    S = r["S_norm"][:]
    T = C.shape[1]
    ptr, fr = r["event_ptr"][:], r["event_frames"][:]
    E = np.zeros(C.shape, bool)
    for i in range(len(C)):
        E[i, fr[ptr[i]:ptr[i + 1]]] = True
    Cn = C / sn[:, None]
    st, en, bl = bouts_of(lab)
    skip = int(round(skip_s * fs))
    cells, pops = [], []
    for k, name in enumerate(names):
        sel = [(a + skip, b) for a, b, l in zip(st, en, bl) if l == k and b - a - skip > 0]
        n_bouts = sum(1 for l in bl if l == k)
        m = np.zeros(T, bool)
        for a, b in sel:
            m[a:b] = True
        secs = m.sum() / fs
        if secs < min_label_s or n_bouts < min_bouts:
            continue
        rate = E[:, m].sum(1) / secs
        trace = Cn[:, m].mean(1)
        s_rate = S[:, m].mean(1) * fs
        base = dict(day=g.name.strip("/"), rec=rec_name, condition=cond, mouse=day_attrs["mouse"],
                    spn=day_attrs["spn"], dose=day_attrs["dose"], label=name)
        cells.append(pd.DataFrame(dict(base, cell=np.arange(len(C)), rate=rate, trace=trace,
                                       s_rate=s_rate, seconds=secs, bouts=n_bouts)))
        pops.append(dict(base, n_cells=len(C), seconds=secs, bouts=n_bouts,
                         rate_median=np.median(rate), rate_mean=rate.mean(),
                         trace_median=np.median(trace), s_rate_median=np.median(s_rate),
                         frac_active=np.mean(rate > 0)))
    return cells, pops

def summary_figure(pop, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [l for l in pop.label.unique()]
    order = ["rest", "walk", "scanning walk", "sharp turn (CW)", "turning run (CCW)",
             "straight run", "turning run (CW)", "wall run"]
    labels = [l for l in order if l in labels] + [l for l in labels if l not in order]
    cols = dict(zip(conditions, ["#2a78d6", "#eda100", "#e87ba4", "#e34948"]))
    readouts = [("rate_median", "event rate (Hz), median over cells"),
                ("trace_median", "denoised trace (noise units), median"),
                ("frac_active", "share of active cells")]
    fig, axes = plt.subplots(len(readouts), 2, figsize=(15, 3.6 * len(readouts)), squeeze=False)
    for j, spn in enumerate(("D1", "D2")):
        d = pop[pop.spn == spn]
        for i, (col, lab) in enumerate(readouts):
            ax = axes[i, j]
            w = 0.8 / len(conditions)
            for c_i, c in enumerate(conditions):
                for l_i, l in enumerate(labels):
                    v = d[(d.condition == c) & (d.label == l)][col].values
                    x = l_i + (c_i - 1.5) * w
                    if len(v):
                        ax.scatter(np.full(len(v), x) + np.random.default_rng(l_i).uniform(-w / 4, w / 4, len(v)),
                                   v, s=6, color=cols[c], alpha=0.5, lw=0)
                        ax.plot([x - w / 2.5, x + w / 2.5], [np.median(v)] * 2, color=cols[c], lw=2)
            ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(lab, fontsize=8)
            ax.set_title(f"{spn} mice: {lab}", fontsize=9, loc="left")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=cols[c], lw=2, label=c) for c in conditions],
               loc="upper center", ncol=4, frameon=False)
    fig.suptitle(f"SPN activity per behaviour label (dots: recordings; bars: median){tag}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{out_dir}/spn_by_label{tag.replace(' ', '_').replace(',', '')}.png", dpi=110)
    plt.close(fig)


def main():
    os.makedirs(out_dir, exist_ok=True)
    cells, pops = [], []
    with h5py.File(dataset, "r") as h:
        for day in h:
            g = h[day]
            for rec in ("rec0", "rec1"):
                c, p = recording_tables(g, dict(g.attrs), rec)
                cells += c
                pops += p
    cell = pd.concat(cells, ignore_index=True)
    pop = pd.DataFrame(pops)
    sfx = f"_skip{skip_s:g}" if skip_s else ""
    cell.to_csv(f"{out_dir}/cell_label_table{sfx}.csv.gz", index=False)
    pop.to_csv(f"{out_dir}/recording_label_table{sfx}.csv", index=False)
    pd.set_option("display.width", 220)
    print(pop.groupby(["spn", "condition", "label"])[["rate_median", "trace_median", "frac_active", "seconds"]]
          .median().round(3).to_string())
    print(pop.groupby(["condition", "label"]).size().unstack(0).to_string())
    summary_figure(pop, f", skip {skip_s:g} s" if skip_s else "")


if __name__ == "__main__":
    main()
