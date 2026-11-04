"""Why do the event rate and the denoised trace disagree across drug conditions? Per cell, the
mean trace decomposes as
    mean(C / sn) = mean(S / sn) x K                      (K = sum C / sum S, calcium kernel integral)
    mean(S / sn) = rate x mass_per_event_window + mass outside event windows
with S the OASIS deconvolved activity, sn the noise level and events the summed detection of the
pipeline (3-frame spike mass >= 1.5 x noise, events >= min_event_sep_s apart). Per cell:
  rate          events / s
  mass3         S / sn summed over the 3-frame detection window, per event
  mass_win      S / sn summed over [event, event + min_event_sep_s), per event (bursts merged)
  S_rate        total S / sn per second
  in_events     share of the S mass inside the event windows
  tau_s         decay time from K (AR(1): g = 1 - 1 / K, tau = -1 / (fs ln g))
  trace         mean C / sn
  sn            noise level (raw fluorescence units)
Per mouse: median over cells; per condition: log10 ratio to the mouse's Control (Wilcoxon over
mice). Both drug cohorts (data files of build_spn_dataset.py).

Outputs (OUT_DIR): cells_trace_vs_spikes.csv.gz, mice_trace_vs_spikes.csv, stats, figure.

  conda run -n parker python analysis_spn_trace_vs_spikes.py
"""
import os
import numpy as np
import pandas as pd
import h5py
from scipy.stats import wilcoxon

data_dir = "/home/rgast/data/parker_data/dataset"
out_dir = os.environ.get("OUT_DIR", "/home/rgast/data/parker_data/spn_trace_vs_spikes")
slide_dir = os.environ.get("SLIDE_DIR", "/home/rgast/OneDrive/presentations/parker/images")
drugs = ["Clozapine", "Haloperidol"]
fs = 20.0
win = int(0.5 * fs)                 # min_event_sep_s of the pipeline
measures = [("rate", "event rate"), ("trace", "mean trace"), ("S_rate", "deconv. mass / s"),
            ("mass3", "mass / event (3 fr.)"), ("mass_win", "mass / event"),
            ("in_events", "mass in events"), ("tau_s", "decay time"), ("sn", "noise level")]

def cells_of(r):
    C, S, sn = r["C"][:], r["S_norm"][:], r["sn"][:]
    ptr, fr = r["event_ptr"][:], r["event_frames"][:]
    n, T = C.shape
    out = []
    for i in range(n):
        ev = fr[ptr[i]:ptr[i + 1]]
        s, c = S[i], C[i] / sn[i]
        tot = s.sum()
        m3 = np.array([s[e:e + 3].sum() for e in ev])
        mw = np.array([s[e:e + win].sum() for e in ev])
        K = c.sum() / tot if tot > 0 else np.nan
        g = 1 - 1 / K if K and K > 1 else np.nan
        out.append(dict(rate=len(ev) / (T / fs), trace=c.mean(), S_rate=tot / (T / fs),
                        mass3=m3.mean() if len(ev) else np.nan, mass_win=mw.mean() if len(ev) else np.nan,
                        in_events=mw.sum() / tot if tot > 0 else np.nan,
                        tau_s=-1 / (fs * np.log(g)) if g and 0 < g < 1 else np.nan, sn=sn[i]))
    return pd.DataFrame(out)

def main():
    os.makedirs(out_dir, exist_ok=True)
    if os.environ.get("FIGURE_ONLY"):
        rt = pd.read_csv(f"{out_dir}/ratios_trace_vs_spikes.csv")
        figure(rt, pd.read_csv(f"{out_dir}/stats_trace_vs_spikes.csv"))
        return
    cells = []
    for drug in drugs:
        conds = ["Control", "Amphetamine", f"{drug} high", f"A + {drug} high"]
        with h5py.File(f"{data_dir}/{drug.lower()}_spn.h5", "r") as h:
            for day in h:
                for rec in ("rec0", "rec1"):
                    r = h[day][rec]
                    if r.attrs["condition"] not in conds:
                        continue
                    c = cells_of(r)
                    c["drug"], c["condition"], c["mouse"], c["spn"], c["day"], c["rec"] = \
                        drug, r.attrs["condition"], h[day].attrs["mouse"], h[day].attrs["spn"], day, rec
                    cells.append(c)
                print(drug, day, flush=True)
    ce = pd.concat(cells, ignore_index=True)
    ce.to_csv(f"{out_dir}/cells_trace_vs_spikes.csv.gz", index=False)
    mi = ce.groupby(["drug", "spn", "mouse", "condition"])[[m for m, _ in measures]].median().reset_index()
    mi.to_csv(f"{out_dir}/mice_trace_vs_spikes.csv", index=False)
    rows = []
    for (drug, spn, mouse), g in mi.groupby(["drug", "spn", "mouse"]):
        c = g[g.condition == "Control"]
        if not len(c):
            continue
        for _, r in g[g.condition != "Control"].iterrows():
            rows.append(dict(drug=drug, spn=spn, mouse=mouse, condition=r.condition,
                             **{m: np.log10(r[m] / c[m].iloc[0]) for m, _ in measures}))
    rt = pd.DataFrame(rows)
    rt.to_csv(f"{out_dir}/ratios_trace_vs_spikes.csv", index=False)
    st = []
    for (drug, cond), g in rt.groupby(["drug", "condition"]):
        for m, _ in measures:
            v = g[m].dropna().values
            st.append(dict(drug=drug, condition=cond, measure=m, n=len(v), mean=v.mean(),
                           pct=100 * (10 ** v.mean() - 1), p=wilcoxon(v).pvalue if len(v) >= 5 else np.nan))
    st = pd.DataFrame(st)
    st.to_csv(f"{out_dir}/stats_trace_vs_spikes.csv", index=False)
    pd.set_option("display.width", 220)
    print(st.pivot_table(index=["drug", "condition"], columns="measure", values="pct").round(0)
          [[m for m, _ in measures]].to_string())
    print(st.pivot_table(index=["drug", "condition"], columns="measure", values="p").round(3)
          [[m for m, _ in measures]].to_string())
    figure(rt, st)

def figure(rt, st):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "axes.edgecolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.linewidth": 0.7, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    col = {"Amphetamine": "#eda100", "high": "#8a4fb5", "A +": "#e34948"}
    colour = lambda c: col["A +"] if c.startswith("A +") else col["Amphetamine"] if c == "Amphetamine" else col["high"]
    show = [m for m in measures if m[0] != "mass3"]
    fig, axes = plt.subplots(len(drugs), len(show), figsize=(7.6, 3.6), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.84, bottom=0.12, wspace=0.12, hspace=0.45)
    for i, drug in enumerate(drugs):
        conds = ["Amphetamine", f"{drug} high", f"A + {drug} high"]
        for j, (m, lab) in enumerate(show):
            ax = axes[i, j]
            for k, c in enumerate(conds):
                v = rt[(rt.drug == drug) & (rt.condition == c)][m].dropna().values
                ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.12, 0.12, len(v)), v,
                           s=5, color=colour(c), alpha=0.45, lw=0)
                ax.errorbar(k, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color=colour(c), lw=1.2)
                ax.scatter([k], [v.mean()], s=22, color=colour(c), edgecolor="white", lw=0.7, zorder=3)
                s_ = st[(st.drug == drug) & (st.condition == c) & (st.measure == m)]
                if len(s_) and s_.p.iloc[0] < 0.05:
                    p = s_.p.iloc[0]
                    ax.text(k, 0.52, "***" if p < 0.001 else "**" if p < 0.01 else "*", ha="center", fontsize=6.5)
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_xticks([])
            ax.set_xlim(-0.6, 2.6)
            ax.set_ylim(-0.6, 0.6)
            ax.yaxis.grid(True, color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(lab, loc="left", fontsize=6.8)
            if j == 0:
                ax.set_ylabel(f"{drug}\nlog$_{{10}}$ ratio to Control")
    hs = [plt.Line2D([], [], color=c, marker="o", lw=0, ms=5) for c in col.values()]
    fig.legend(hs, ["Amph.", "drug alone (high)", "Amph. + drug (high)"], loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.08, 0.02, "dots: mice (median over cells, ratio to the same mouse's Control); "
             "mean trace = deconvolved mass / s x calcium kernel (decay time)", fontsize=6.3, color=INK2)
    for d in (out_dir, slide_dir):
        fig.savefig(f"{d}/spn_trace_vs_spikes.pdf")
        fig.savefig(f"{d}/spn_trace_vs_spikes.png", dpi=200)
    plt.close(fig)
    print("saved spn_trace_vs_spikes.pdf/.png")


if __name__ == "__main__":
    main()
