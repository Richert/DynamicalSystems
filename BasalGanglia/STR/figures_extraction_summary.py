"""Summary figures of the final calcium extraction per drug (pipeline_raw_spikes.py, full runs):
extraction quality per recording and behaviour-independent firing-rate distributions.

  extraction_quality   per recording (15-min / 45-min): cells, share of the original pipeline's
                       active neurons recovered, matched-trace correlation, recall of the original
                       events, components removed per selection rule, median event rate
  extraction_rates     per-cell whole-session event rates, D1 / D2 mice, per condition (Control,
                       Amphetamine, drug alone, Amph. + drug; high dose), with mouse medians

Inputs: local copies of the evaluation summaries (summary_<tag>_thr1.5_sum3_<sel>.json) and the
event matrices (spikes_<...>.npy, cells x frames) of each recording; mouse -> D1 / D2 from the
QC table.

  conda run -n parker python figures_extraction_summary.py [Drug ...]   (default: Clozapine Haloperidol)
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

local = "/home/rgast/data/parker_data/raw_pipeline"
qc_csv = "/mnt/kennedy_labdata/Parkerlab/results/raw_pipeline/qc_all_recordings.csv"
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/extraction_summary")
tag, sel = "p1_c0.6_pnr5_nrg0.999", "snr2_sz_rim5_hyb0.3-0.2_ves3_off6_fedge1.05"
fs = 20.0
drug_codes = {"Clozapine": "clo", "Haloperidol": "hal"}
dose_of = {"01": "Vehicle", "05": "LowDose", "03": "LowDose", "04": "HighDose"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DRUG_COL = {"Clozapine": "#2a78d6", "Haloperidol": "#1f8a70"}
RULES = [("n_size_rel_dropped", "oversized"), ("n_lens_rim_dropped", "lens rim"),
         ("n_hybrid_consistency_dropped", "inconsistent"), ("n_vessel_dropped", "blood vessel"),
         ("n_off_peak_dropped", "off-peak"), ("n_edge_piece_dropped", "edge piece")]
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.7,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

def save(fig, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.pdf")
    fig.savefig(f"{out_dir}/{name}.png", dpi=200)
    plt.close(fig)
    print(f"saved {out_dir}/{name}.pdf/.png")

def spines(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

def load(drugs):
    spn = dict(pd.read_csv(qc_csv)[["mouse", "spn"]].drop_duplicates().values)
    recs, cells = [], []
    for drug in drugs:
        code = drug_codes[drug]
        for d in sorted(glob.glob(f"{local}/*_{code}0?_amph_rec?")):
            f = f"{d}/summary_{tag}_thr1.5_sum3_{sel}.json"
            if not os.path.isfile(f):
                continue
            name = os.path.basename(d)
            s = json.load(open(f))
            parts = name.split("_")
            mouse = parts[1].replace("m85", "m085")
            dose = dose_of[parts[2][-2:]]
            rec = int(name[-1])
            cond = {("Vehicle", 0): "Control", ("Vehicle", 1): "Amphetamine",
                    ("LowDose", 0): f"{drug} low", ("LowDose", 1): f"A + {drug} low",
                    ("HighDose", 0): f"{drug} high", ("HighDose", 1): f"A + {drug} high"}[(dose, rec)]
            row = dict(drug=drug, recording=name, mouse=mouse, spn=spn.get(mouse, ""), dose=dose,
                       rec=rec, condition=cond, n_cells=s["n_new"],
                       recovered=s.get("frac_active_old_matched", np.nan),
                       r_matched=s.get("r_matched_median", np.nan), recall=s.get("recall_mean", np.nan),
                       noise_fp=s.get("noise_fp_share", np.nan),
                       **{k: s.get(k, 0) or 0 for k, _ in RULES})
            ev = np.load(f"{d}/spikes_{tag}_thr1.5_sum3_{sel}.npy")
            T = ev.shape[1] / fs
            rate = ev.sum(1) / T
            row.update(minutes=T / 60, rate_median=np.median(rate))
            recs.append(row)
            cells.append(pd.DataFrame(dict(drug=drug, recording=name, mouse=mouse, spn=row["spn"],
                                           condition=cond, rate=rate,
                                           log_rate=np.log10(np.maximum(rate, 0.5 / T)))))
    return pd.DataFrame(recs), pd.concat(cells, ignore_index=True)

def fig_quality(rd, drugs):
    metrics = [("n_cells", "cells per recording", None), ("recovered", "original active\nneurons recovered", (0, 1.02)),
               ("r_matched", "matched-trace\ncorrelation (median)", (0, 1.02)),
               ("recall", "recall of the\noriginal events", (0, 1.02)), ("rate_median", "median event\nrate (Hz)", None)]
    fig = plt.figure(figsize=(6.0, 3.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05], hspace=0.75, wspace=0.75, left=0.075,
                          right=0.99, top=0.88, bottom=0.12)
    rng = np.random.default_rng(1)
    for i, (col, lab, lim) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        for j, drug in enumerate(drugs):
            for k, rec in enumerate((0, 1)):
                v = rd[(rd.drug == drug) & (rd.rec == rec)][col].dropna().values
                x = j * 2.4 + k
                ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color=DRUG_COL[drug],
                           alpha=0.4 if rec else 0.65, lw=0, marker="o" if rec == 0 else "s")
                ax.plot([x - 0.3, x + 0.3], [np.median(v)] * 2, color=INK, lw=1.4)
        ax.set_xticks([j * 2.4 + k for j in range(len(drugs)) for k in (0, 1)],
                      ["15'", "45'"] * len(drugs), fontsize=6)
        for j, drug in enumerate(drugs):
            ax.text(j * 2.4 + 0.5, -0.24, drug[:3] + ".", transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=6.5, color=DRUG_COL[drug], fontweight="bold")
        if lim:
            ax.set_ylim(*lim)
        ax.set_title(lab, loc="left", fontsize=7.2)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        spines(ax)
    # components removed per selection rule (mean per recording), per drug
    ax = fig.add_subplot(gs[1, :3])
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.07, pos.y0, pos.width - 0.07, pos.height])
    w = 0.38
    for j, drug in enumerate(drugs):
        m = [rd[rd.drug == drug][k].mean() for k, _ in RULES]
        y = np.arange(len(RULES)) + (j - 0.5) * w
        ax.barh(y, m, height=w, color=DRUG_COL[drug], label=drug)
        for yi, v in zip(y, m):
            ax.text(v + 0.4, yi, f"{v:.0f}", va="center", fontsize=5.8, color=DRUG_COL[drug])
    ax.set_yticks(np.arange(len(RULES)), [l for _, l in RULES], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("components removed per recording (mean)")
    ax.set_title("selection rules after CNMF-E", loc="left", fontsize=7.5)
    ax.legend(frameon=False, loc="lower right")
    spines(ax)
    # recordings / mice / cells
    ax = fig.add_subplot(gs[1, 3:])
    ax.axis("off")
    lines = []
    for drug in drugs:
        d = rd[rd.drug == drug]
        lines.append((drug, f"{len(d)} recordings, {d.mouse.nunique()} mice\n"
                            f"{d.n_cells.sum():,} cells ({d.n_cells.median():.0f} per recording)\n"
                            f"doses: {', '.join(sorted(d.dose.unique()))}".replace(",", " ", 1)))
    for i, (drug, t) in enumerate(lines):
        ax.text(0.08, 1.0 - i * 0.55, drug, fontsize=8.5, fontweight="bold", color=DRUG_COL[drug],
                va="top", transform=ax.transAxes)
        ax.text(0.08, 0.86 - i * 0.55, t, fontsize=6.8, color=INK2, va="top", transform=ax.transAxes,
                linespacing=1.35)
    fig.text(0.075, 0.965, "Extraction quality per recording (bars: median; circles: 15-min, squares: "
             "45-min recordings)", fontsize=8.5, color=INK)
    save(fig, "extraction_quality")

def fig_rates(cl, drugs):
    fig, axes = plt.subplots(len(drugs), 4, figsize=(6.0, 1.85 * len(drugs) + 0.5),
                             gridspec_kw=dict(width_ratios=[1.6, 0.55, 1.6, 0.55]), squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.12, wspace=0.42, hspace=0.6)
    med = cl.groupby(["drug", "spn", "mouse", "condition"]).log_rate.median()
    mlo, mhi = med.min() - 0.05, med.max() + 0.05
    bins = np.linspace(-2.6, 0.4, 31)
    for i, drug in enumerate(drugs):
        conds = ["Control", "Amphetamine", f"{drug} high", f"A + {drug} high"]
        col = dict(zip(conds, ["#2a78d6", "#eda100", "#8a4fb5", "#e34948"]))
        short = dict(zip(conds, ["Control", "Amph.", f"{drug[:4]}. high", f"Amph. + {drug[:4]}. high"]))
        for g, spn in enumerate(("D1", "D2")):
            ax = axes[i, 2 * g]
            d = cl[(cl.drug == drug) & (cl.spn == spn)]
            for c in conds:
                v = d[d.condition == c].log_rate.clip(bins[0], bins[-1])
                if len(v):
                    h, _ = np.histogram(v, bins=bins)
                    ax.stairs(h / h.sum(), bins, color=col[c], lw=1.3, label=short[c])
            ax.set_xticks(np.log10([0.01, 0.1, 1]), ["0.01", "0.1", "1"])
            ax.set_xlim(bins[0], bins[-1])
            if i == len(drugs) - 1:
                ax.set_xlabel("event rate per cell (Hz)")
            ax.set_title(f"{drug}: {spn} mice", loc="left", fontsize=8, color=DRUG_COL[drug],
                         fontweight="bold")
            ax.text(0.98, 0.95, f"{len(d):,} cells\n{d.mouse.nunique()} mice".replace(",", " "),
                    transform=ax.transAxes, ha="right", va="top", fontsize=6, color=INK2)
            if g == 0:
                ax.set_ylabel("share of cells")
            spines(ax)
            ax = axes[i, 2 * g + 1]
            p = d.groupby(["mouse", "condition"]).log_rate.median().unstack().reindex(columns=conds)
            for _, v in p.iterrows():
                ok = v.notna().values
                ax.plot(np.arange(4)[ok], v.values[ok], color=GRID, lw=0.5, zorder=1)
            for ci, c in enumerate(conds):
                v = p[c].dropna().values
                ax.scatter(np.full(len(v), ci), v, s=6, color=col[c], alpha=0.6, lw=0, zorder=2)
                if len(v):
                    ax.scatter([ci], [np.median(v)], s=22, color=col[c], edgecolor="white", lw=0.7, zorder=3)
            yt = [t for t in (0.02, 0.03, 0.05, 0.1, 0.2) if mlo <= np.log10(t) <= mhi]
            ax.set_yticks(np.log10(yt), [f"{t:g}" for t in yt])
            ax.set_ylim(mlo, mhi)
            ax.set_xticks([])
            ax.set_xlim(-0.6, 3.6)
            ax.set_title("mice (Hz)", loc="left", fontsize=6, color=INK2)
            ax.tick_params(axis="y", labelsize=6)
            spines(ax)
    fig.text(0.1, 0.965, "Whole-session firing rates (independent of behaviour), high dose", fontsize=8.5,
             color=INK)
    hs = [plt.Line2D([], [], color=c, lw=1.5) for c in ("#2a78d6", "#eda100", "#8a4fb5", "#e34948")]
    fig.legend(hs, ["Control", "Amph.", "drug alone", "Amph. + drug"], loc="upper right", ncol=4,
               frameon=False, bbox_to_anchor=(0.99, 0.945), fontsize=6.8, handlelength=1.4)
    save(fig, "extraction_rates")


if __name__ == "__main__":
    drugs = sys.argv[1:] or ["Clozapine", "Haloperidol"]
    rd, cl = load(drugs)
    fig_quality(rd, drugs)
    fig_rates(cl, drugs)
    rd.to_csv(f"{out_dir}/recordings.csv", index=False)
    pd.set_option("display.width", 200)
    print(rd.groupby(["drug", "rec"])[["n_cells", "recovered", "r_matched", "recall", "rate_median"]]
          .median().round(3).to_string())
    print(cl.groupby(["drug", "spn", "condition"]).log_rate.median().round(3).unstack(0).to_string())
