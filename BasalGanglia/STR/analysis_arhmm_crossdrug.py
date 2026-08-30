"""Cross-drug comparison of SPN dynamics in the A + Drug condition.

Consumes the per-(session, state) readouts written by analysis_arhmm_spn_states.py
(`arhmm_spn_state_readouts.csv`), which are defined on SHARED AR-HMM states, so
states are directly comparable across drugs. This script asks, *within the
A + Drug condition only*: does drug identity modulate the SPN firing-rate CDF
centre / slope / dimensionality, and does that modulation depend on the
behavioural (speed) state?

  - value ~ C(drug) * C(state) + (1|animal)   [mixed model; drug is between-animal]
  - per-state pairwise drug contrasts (Welch t on per-animal values), FDR-corrected

Run with the dedicated env:
  /home/rgast/conda/envs/arhmm/bin/python analysis_arhmm_crossdrug.py
"""

import pickle
import numpy as np
from pandas import read_csv
from itertools import combinations
from scipy.stats import ttest_ind

save_dir = "/home/rgast/data/parker_data"
csv_path = f"{save_dir}/arhmm_spn_state_readouts.csv"
results_pkl = f"{save_dir}/arhmm_spn_results.pkl"

target_condition = "A + Drug"
quantities = ["center", "slope", "dim"]
min_animals_per_cell = 3       # need this many animals in a (drug, state) cell
drugs_order = ["haloperidol", "xanomeline", "MP10"]

# ---------------------------------------------------------------------------

def per_animal_table(df, q):
    """One value per (drug, mouse, state): average over that animal's sessions."""
    d = df.dropna(subset=[q])
    d = d[d["condition"] == target_condition]
    return d.groupby(["drug", "mouse", "state"], as_index=False)[q].mean()

def mixed_model(d, q):
    """value ~ C(drug)*C(state) + (1|mouse), with additive / clustered-OLS fallbacks."""
    import statsmodels.formula.api as smf
    # keep (drug,state) cells with enough animals, then states/drugs present in >=2 of the other
    cell = d.groupby(["drug", "state"])["mouse"].nunique()
    d = d[d.set_index(["drug", "state"]).index.isin(cell[cell >= min_animals_per_cell].index)].copy()
    for _ in range(3):
        ds = d.groupby("state")["drug"].nunique(); d = d[d["state"].isin(ds[ds >= 2].index)]
        sd = d.groupby("drug")["state"].nunique(); d = d[d["drug"].isin(sd[sd >= 2].index)]
    if d["drug"].nunique() < 2 or d["state"].nunique() < 2 or len(d) < 8:
        return None, d
    d["state"] = d["state"].astype("category")
    for formula, kind in [(f"{q} ~ C(drug)*C(state)", "mixedlm-interaction"),
                          (f"{q} ~ C(drug)+C(state)", "mixedlm-additive")]:
        try:
            mdf = smf.mixedlm(formula, d, groups=d["mouse"]).fit(reml=False, method="lbfgs")
            if np.all(np.isfinite(mdf.pvalues.values)):
                return (mdf, kind), d
        except Exception:
            pass
    try:
        mdf = smf.ols(f"{q} ~ C(drug)+C(state)", d).fit(
            cov_type="cluster", cov_kwds={"groups": d["mouse"]})
        return (mdf, "ols-clustered-additive"), d
    except Exception as e:
        return ("error", str(e)), d

def pairwise_by_state(d, q):
    """Welch t between drug pairs within each state (per-animal values)."""
    from statsmodels.stats.multitest import multipletests
    rows = []
    for state, g in d.groupby("state"):
        present = [dr for dr in drugs_order if dr in g["drug"].unique()]
        for a, b in combinations(present, 2):
            xa = g[g["drug"] == a][q].values
            xb = g[g["drug"] == b][q].values
            if len(xa) >= min_animals_per_cell and len(xb) >= min_animals_per_cell:
                t, p = ttest_ind(xa, xb, equal_var=False)
                rows.append(dict(state=int(state), pair=f"{a} vs {b}",
                                 mean_a=xa.mean(), mean_b=xb.mean(), p=p))
    if rows:
        ps = [r["p"] for r in rows]
        _, q_adj, *_ = multipletests(ps, method="fdr_bh")
        for r, qa in zip(rows, q_adj):
            r["q_fdr"] = qa
    return rows

# ---------------------------------------------------------------------------

def main():
    df = read_csv(csv_path)
    try:
        state_speed = pickle.load(open(results_pkl, "rb"))["state_speed"]
    except Exception:
        state_speed = None

    print(f"A + Drug cross-drug comparison  (drugs present: "
          f"{sorted(set(df[df['condition']==target_condition]['drug']))})")

    summary = {}
    for q in quantities:
        d = per_animal_table(df, q)
        print(f"\n================  {q}  ================")
        (model, d_used) = mixed_model(d, q)
        if model is None:
            print("  insufficient data for a model")
        elif model[0] == "error":
            print("  model failed:", model[1])
        else:
            mdf, kind = model
            print(f"  [{kind}]  fixed effects involving drug:")
            for term in mdf.pvalues.index:
                if "drug" in term.lower():
                    print(f"    {term:34s} p={mdf.pvalues[term]:.4g}")
        pw = pairwise_by_state(d, q)
        if pw:
            print("  per-state pairwise drug contrasts (Welch t, FDR):")
            for r in pw:
                star = "*" if r["q_fdr"] < 0.05 else ""
                print(f"    S{r['state']}  {r['pair']:26s} "
                      f"{r['mean_a']:.3f} vs {r['mean_b']:.3f}  q={r['q_fdr']:.3g} {star}")
        summary[q] = dict(pairwise=pw)

    plot(df, state_speed)
    pickle.dump(summary, open(f"{save_dir}/arhmm_crossdrug_Adrug.pkl", "wb"))
    print(f"\nSaved cross-drug summary + figure to {save_dir}/")

def plot(df, state_speed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sb
    d = df[df["condition"] == target_condition]
    drugs_present = [dr for dr in drugs_order if dr in d["drug"].unique()]
    pal = dict(zip(drugs_present, sb.color_palette("Set2", len(drugs_present))))

    fig, axes = plt.subplots(1, len(quantities), figsize=(5 * len(quantities), 4.2), layout="tight")
    for ax, q in zip(np.atleast_1d(axes), quantities):
        dd = d.dropna(subset=[q])
        # per-animal values, then mean +/- sem across animals per (drug, state)
        pa = dd.groupby(["drug", "mouse", "state"], as_index=False)[q].mean()
        g = pa.groupby(["drug", "state"])[q].agg(["mean", "sem"])
        for dr in drugs_present:
            if dr in g.index.get_level_values("drug"):
                sub = g.xs(dr, level="drug")
                ax.errorbar(sub.index, sub["mean"], yerr=sub["sem"], marker="o",
                            color=pal[dr], label=dr, capsize=3)
        ax.set(xlabel="state (speed-ordered)", ylabel=q,
               title=f"A + Drug: SPN {q} across drugs")
    np.atleast_1d(axes)[0].legend(title="drug", fontsize=9)
    fig.suptitle("Cross-drug comparison in the A + Drug condition (shared AR-HMM states)",
                 fontsize=13)
    fig.savefig(f"{save_dir}/arhmm_crossdrug_Adrug.png", dpi=120)
    print(f"Saved figure to {save_dir}/arhmm_crossdrug_Adrug.png")


if __name__ == "__main__":
    main()
