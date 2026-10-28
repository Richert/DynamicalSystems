"""Cross-drug comparison of drug effects on SPN firing-rate statistics.

Consumes the per-(session, state) readouts written by analysis_arhmm_spn_states.py
(or the embedding pipeline), which are defined on SHARED behavioural states, so
states are directly comparable across drugs. Every drug has its own animal cohort
with its own Control/Amphetamine sessions, so drugs are compared on WITHIN-ANIMAL
changes per (mouse, state) rather than on raw values:

  drug       = Drug     - Control       (effect of the drug alone)
  drug_amph  = A + Drug - Amphetamine   (effect of the drug on top of amphetamine)

For each contrast, SPN type (D1/D2 analysed separately) and readout:
  - delta ~ C(drug) * C(state) + (1|mouse)   [mice are partly shared across drug
    cohorts -> random intercept; reference drug = ref_drug]
  - per state: delta ~ C(drug) + (1|mouse) and all pairwise drug contrasts (Wald,
    from that model), BH-FDR across all contrasts of a readout
  - per drug and state: mean +/- sem of the per-mouse deltas

Run with the dedicated env:
  /home/rgast/conda/envs/arhmm/bin/python analysis_arhmm_crossdrug.py
"""

import os
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import norm
import analysis_arhmm_spn_states as A   # mixed-model fitting helpers

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

run_dir = f"{A.save_dir}/arhmm_all10/K3"        # output folder of the state analysis
prefix = "arhmm"                                 # "arhmm" | "embed"
out_dir = f"{run_dir}/crossdrug"

# drugs to compare (MP10: PDE10A inhibitor, SKF38393: D1 agonist, olanzapine: D2>D1
# antagonist, SCH23390: D1 antagonist, haloperidol: D2 antagonist = reference)
drugs = ["haloperidol", "MP10", "SKF38393", "olanzapine", "SCH23390"]
ref_drug = "haloperidol"
contrasts = {"drug": ("Drug", "Control"), "drug_amph": ("A + Drug", "Amphetamine")}
quantities = ["mean_log", "sd_log", "corr_active"]
spn_types = ["D1", "D2"]
min_animals_per_cell = 3       # need this many animals in a (drug, state) cell

# ---------------------------------------------------------------------------
# within-animal deltas
# ---------------------------------------------------------------------------

def deltas(df, q, cond, ref):
    """Per (drug, spn, mouse, state): mean over sessions of q in `cond` minus that
    in `ref` (both from the same drug cohort); animals need both."""
    pa = df.dropna(subset=[q]).groupby(["drug", "spn", "mouse", "state", "condition"])[q].mean()
    pa = pa.unstack("condition")
    if cond not in pa or ref not in pa:
        return pd.DataFrame(columns=["drug", "spn", "mouse", "state", "delta"])
    d = (pa[cond] - pa[ref]).dropna().rename("delta").reset_index()
    return d

def keep_cells(d):
    """Keep (drug, state) cells with enough animals and drugs/states present in >= 2
    levels of the other factor."""
    cell = d.groupby(["drug", "state"])["mouse"].nunique()
    d = d[d.set_index(["drug", "state"]).index.isin(cell[cell >= min_animals_per_cell].index)]
    for _ in range(3):
        ds = d.groupby("state")["drug"].nunique(); d = d[d["state"].isin(ds[ds >= 2].index)]
        sd = d.groupby("drug")["state"].nunique(); d = d[d["drug"].isin(sd[sd >= 2].index)]
    return d.copy()

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def fit(d, rhs):
    """Mixed model delta ~ rhs + (1|mouse) on a full-rank design (empty-cell columns
    dropped); falls back to OLS with mouse-clustered SEs. Returns (model, kind) or
    (None, reason)."""
    import statsmodels.api as sm
    from patsy import dmatrices
    y, X = dmatrices(f"delta ~ {rhs}", d, return_type="dataframe")
    X = X.loc[:, (X != 0).any(axis=0)]
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        return None, "rank-deficient design"
    n_mice = d["mouse"].nunique()
    for kind in ["mixedlm", "ols-clustered"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if kind == "mixedlm":
                    m = A._fit_mixedlm(y, X, d["mouse"].values)
                else:
                    m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={
                        "groups": d["mouse"].astype("category").cat.codes.values})
            ok, why = A._valid_fit(m, kind, n_mice)
            if ok:
                return m, kind
        except Exception as e:
            why = str(e)
    return None, why

def set_levels(d):
    drugs_present = [dr for dr in drugs if dr in set(d["drug"])]
    if ref_drug in drugs_present:
        drugs_present = [ref_drug] + [dr for dr in drugs_present if dr != ref_drug]
    d["drug"] = pd.Categorical(d["drug"], categories=drugs_present)
    d["state"] = pd.Categorical(d["state"], categories=sorted(d["state"].unique()))
    return d

def omnibus(d):
    """delta ~ C(drug)*C(state) + (1|mouse), additive fallback. Returns term rows."""
    for design, rhs in [("interaction", "C(drug)*C(state)"), ("additive", "C(drug)+C(state)")]:
        m, kind = fit(d, rhs)
        if m is not None:
            p = m.pvalues.drop(labels=[i for i in m.pvalues.index if i in ("Intercept", "Group Var")],
                               errors="ignore")
            return [dict(term=t, coef=float(m.params[t]), se=float(m.bse[t]), p=float(p[t]),
                         model=f"{kind}-{design}") for t in p.index]
    return []

def pairwise_by_state(d):
    """Per state: delta ~ C(drug) + (1|mouse); Wald contrasts for every drug pair."""
    rows = []
    for state, g in d.groupby("state", observed=True):
        g = g.copy()
        g["drug"] = g["drug"].cat.remove_unused_categories()
        if g["drug"].nunique() < 2:
            continue
        m, kind = fit(g, "C(drug)")
        if m is None:
            continue
        levels = list(g["drug"].cat.categories)
        p, C = m.params, m.cov_params()
        def vec(dr):               # contrast vector for the mean of drug `dr`
            v = pd.Series(0.0, index=p.index)
            v["Intercept"] = 1.0
            if dr != levels[0]:
                v[f"C(drug)[T.{dr}]"] = 1.0
            return v
        for a, b in combinations(levels, 2):
            c = vec(b) - vec(a)
            est = float(c @ p)
            se = float(np.sqrt(c @ C.loc[p.index, p.index] @ c))
            rows.append(dict(state=int(state), pair=f"{b} - {a}", diff=est, se=se,
                             p=2 * norm.sf(abs(est / se)), model=kind))
    return rows

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    from statsmodels.stats.multitest import multipletests
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(f"{run_dir}/{prefix}_spn_state_readouts.csv")
    df = df[df["drug"].isin(drugs)]
    print(f"Cross-drug comparison on {run_dir} ({prefix}); drugs: {drugs}, reference: {ref_drug}")

    all_deltas, terms, pairs = [], [], []
    for cname, (cond, ref) in contrasts.items():
        for spn in spn_types:
            for q in quantities:
                d = deltas(df[df["spn"] == spn], q, cond, ref)
                if d.empty:
                    continue
                d = d.assign(contrast=cname, quantity=q)
                all_deltas.append(d)
                dk = keep_cells(d)
                if dk["drug"].nunique() < 2 or len(dk) < 8:
                    print(f"  [{cname} {spn} {q}] insufficient data")
                    continue
                dk = set_levels(dk)
                for r in omnibus(dk):
                    terms.append(dict(contrast=cname, spn=spn, quantity=q, **r))
                for r in pairwise_by_state(dk):
                    pairs.append(dict(contrast=cname, spn=spn, quantity=q, **r))

    deltas_df = pd.concat(all_deltas)
    terms_df, pairs_df = pd.DataFrame(terms), pd.DataFrame(pairs)
    for t in (terms_df, pairs_df):        # BH-FDR within (contrast, spn, quantity)
        t["q_fdr"] = np.nan
        for _, g in t.groupby(["contrast", "spn", "quantity"]):
            t.loc[g.index, "q_fdr"] = multipletests(g["p"], method="fdr_bh")[1]
    summary = (deltas_df.groupby(["contrast", "spn", "quantity", "drug", "state"])["delta"]
               .agg(["mean", "sem", "count"]).reset_index())

    deltas_df.to_csv(f"{out_dir}/crossdrug_deltas.csv", index=False)
    summary.to_csv(f"{out_dir}/crossdrug_delta_summary.csv", index=False)
    terms_df.to_csv(f"{out_dir}/crossdrug_model_terms.csv", index=False)
    pairs_df.to_csv(f"{out_dir}/crossdrug_pairwise.csv", index=False)

    pd.set_option("display.width", 220)
    for (cname, spn, q), g in terms_df.groupby(["contrast", "spn", "quantity"]):
        sig = g[(g["q_fdr"] < 0.05) & g["term"].str.contains("drug")]
        print(f"\n=== {cname} | {spn} | {q} [{g['model'].iloc[0]}]: "
              f"{len(sig)} significant drug terms (ref = {ref_drug})")
        if len(sig):
            print(sig[["term", "coef", "q_fdr"]].round(4).to_string(index=False))
    sigp = pairs_df[pairs_df["q_fdr"] < 0.05]
    print(f"\nsignificant pairwise drug contrasts per state (q<0.05): {len(sigp)} of {len(pairs_df)}")
    if len(sigp):
        print(sigp[["contrast", "spn", "quantity", "state", "pair", "diff", "q_fdr"]]
              .round(4).to_string(index=False))
    if A.plot_results:
        plot(summary)
    print(f"\nSaved cross-drug tables and figure to {out_dir}/")

def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sb
    pal = dict(zip(drugs, sb.color_palette("colorblind", len(drugs))))
    rows = [(c, s) for c in contrasts for s in spn_types]
    fig, axes = plt.subplots(len(rows), len(quantities), figsize=(5 * len(quantities), 3.4 * len(rows)),
                             layout="tight", squeeze=False)
    for i, (cname, spn) in enumerate(rows):
        for j, q in enumerate(quantities):
            ax = axes[i, j]
            g = summary[(summary.contrast == cname) & (summary.spn == spn) & (summary.quantity == q)
                        & (summary["count"] >= min_animals_per_cell)]
            for k, dr in enumerate(drugs):
                h = g[g.drug == dr]
                ax.errorbar(h.state + (k - (len(drugs) - 1) / 2) * 0.07, h["mean"], yerr=h["sem"],
                            marker="o", capsize=2, color=pal[dr], label=dr)
            ax.axhline(0, c="0.6", lw=1)
            cond, ref = contrasts[cname]
            ax.set(title=f"{spn}: {q}", xlabel="state (speed-ordered)",
                   ylabel=f"{cond} - {ref}")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Cross-drug comparison of within-animal drug effects ({prefix} states)", fontsize=13)
    fig.savefig(f"{out_dir}/crossdrug_deltas.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
