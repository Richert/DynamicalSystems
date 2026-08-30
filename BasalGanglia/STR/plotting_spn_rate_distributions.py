import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
from scipy.special import erf
from scipy.stats import cramervonmises, cramervonmises_2samp

# preparations
##############

# per-neuron firing rates extracted via OASIS deconvolution (produced by
# extract_oasis_rates.py). Rates are the whole-session mean of the OASIS
# instantaneous rate (s * fs), uncalibrated (arbitrary units).
path = f"/home/rgast/data/parker_data"
h5_path = f"{path}/oasis_rates.h5"

# choose which drugs to show (one row per drug -> n rows)
drugs = [
    "haloperidol", "xanomeline", "MP10",
    # "SCH39166", "SKF38393"
]

# SPN population analysed: "D1", "D2", or "all" (D1 + D2 combined)
cell_type = "D1"
spn_titles = {"D1": "D1-SPNs", "D2": "D2-SPNs", "all": "all SPNs"}

# drug conditions overlaid within each panel (color-coded)
conditions = ["Control", "Amphetamine", "A + Drug"]

# condition pairs tested against each other
comparisons = [("Control", "Amphetamine"), ("Amphetamine", "A + Drug")]

# number of resamples for the permutation (center) and bootstrap (slope) tests
n_perm = 2000
n_boot = 1000

# --- output sizing/style: fit into panel B of project_1.svg. The figure is the
# full page width and its axes spines are pinned to match panel C's axes box
# exactly (measured from project_1.svg), so placing B at x=0 makes B's left and
# right axis lines coincide with C's. Embedded ~1:1 (matplotlib pt == SVG pt);
# the rest of that figure uses font.size ~16, so we match it here. ---
page_width_pt = 989.276   # project_1.svg page width
left_spine_pt = 72.0      # panel C left axis line (SVG x)
right_spine_pt = 988.5    # panel C right axis line (SVG x)
fig_width_pt = page_width_pt
fig_height_pt = 235.0     # panel-B height (previous 196 * ~1.2)
margin_top = 0.86         # axes-area top    (figure fraction)
margin_bottom = 0.27      # axes-area bottom (room for x-label + ticks)
subplot_wspace = 0.22     # horizontal gap between the 3 subplots
base_fontsize = 16.0      # matches panel C/D text
annot_fontsize = 8.5      # compact stats annotation for the short panel
show_suptitle = False     # panel gets its own caption in the proposal

# D1/D2 mouse identity (from analysis_behavior_spn_relationship.py)
mice = {"D1": ["m085", "m040", "m298", "m404", "f487", "f694", "f857", "f859", "m794", "m797",
               "m795", "m973", "m974", "m659", "m975", "f976", "f977", "f979"],
        "D2": ["m971", "m972", "m106", "m120", "m377", "m380", "f414", "f480", "m483", "m485",
               "m241", "m242", "m523", "f605", "f808", "f811", "f840"]}
d1_set, d2_set = set(mice["D1"]), set(mice["D2"])

def spn_type(mouse: str):
    return "D1" if mouse in d1_set else ("D2" if mouse in d2_set else None)

def condition_label(dose: str, cond: str):
    """Map (dose, condition) to the plotted label, mirroring the original
    analysis. Vehicle-dose veh/amph give Control/Amphetamine, HighDose+amph
    gives 'A + Drug'; low dose and the Low/High-dose vehicle runs are dropped."""
    if dose == "Vehicle" and cond == "veh":
        return "Control"
    if dose == "Vehicle" and cond == "amph":
        return "Amphetamine"
    if dose == "HighDose" and cond == "amph":
        return "A + Drug"
    return None

# load per-neuron whole-session mean rates for every relevant session
records = {"drug": [], "condition": [], "SPNs": [], "rates": []}
with h5py.File(h5_path, "r") as h5:
    def collect(name, obj):
        if not (isinstance(obj, h5py.Dataset) and name.endswith("rates")):
            return
        a = obj.parent.attrs
        c, spn = condition_label(str(a["dose"]), str(a["condition"])), spn_type(str(a["mouse"]))
        if c is None or spn is None:
            return
        records["drug"].append(str(a["drug"]))
        records["condition"].append(c)
        records["SPNs"].append(spn)
        records["rates"].append(obj[...].mean(axis=1))  # per-neuron mean rate
    h5.visititems(collect)
for k in ("drug", "condition", "SPNs"):
    records[k] = np.asarray(records[k])

# data / plotting parameters. Rates are winsorized into rate_lims and shown on a
# log x-axis: mass above the upper cap piles at the right edge (so every curve
# reaches 1), and the lower floor lets the log axis display near-zero / silent
# neurons (~1% of rates are exactly 0 and cannot appear on a log scale).
rate_lims = (1e-6, 1e-3)  # (display floor, winsorization cap) for OASIS rate units

def ecdf(x: np.ndarray) -> tuple:
    """Empirical cumulative distribution function of the winsorized rates:
    sorted values and their cumulative probability (spans 0 to 1)."""
    xs = np.sort(np.clip(x, *rate_lims))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    # prepend the floor at y=0 so the curve starts on the axis
    return np.concatenate([[rate_lims[0]], xs]), np.concatenate([[0.0], ys])

def get_rates(drug: str, cond: str, spn: str) -> np.ndarray:
    """Concatenate the per-neuron mean rates of every matching session into a
    single sample (pools all sessions/mice of the requested SPN type)."""
    idx = (records["drug"] == drug) & (records["condition"] == cond)
    if spn != "all":
        idx = idx & (records["SPNs"] == spn)
    sel = np.where(idx)[0]
    if len(sel) == 0:
        return np.asarray([])
    return np.concatenate([records["rates"][i] for i in sel])

# ---------------------------------------------------------------------------
# statistics: ERF (Gaussian-CDF) fits + center / slope tests
# ---------------------------------------------------------------------------

def to_lograte(rates: np.ndarray) -> np.ndarray:
    """Winsorized log10 rate -- the variable the ECDFs are drawn against."""
    return np.log10(np.clip(rates, *rate_lims))

def erf_cdf(x, mu, sigma):
    """ERF sigmoid = Gaussian CDF with center mu and slope 1/sigma."""
    return 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))

def fit_erf(rates: np.ndarray) -> tuple:
    """MLE ERF fit to log-rates: returns (center mu, sigma)."""
    x = to_lograte(rates)
    return float(np.mean(x)), float(np.std(x))

def center_test(rates_a: np.ndarray, rates_b: np.ndarray, rng) -> float:
    """Permutation test for a difference in CDF center (mean log-rate).
    Returns the p_value."""
    xa, xb = to_lograte(rates_a), to_lograte(rates_b)
    obs = xb.mean() - xa.mean()
    pooled = np.concatenate([xa, xb])
    na = len(xa)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(perm[na:].mean() - perm[:na].mean()) >= abs(obs) - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)

def slope_test(rates_ref: np.ndarray, rates_test: np.ndarray, rng) -> float:
    """Selective-parameter-fixing test (the requested ERF + CvM approach):
    fix the test condition's slope to the reference's fitted slope, fit only
    its center, and test the goodness-of-fit of that shift-only ERF model to
    the test data via the one-sample Cramer-von Mises statistic. Because the
    center is estimated, the null is obtained by parametric bootstrap. A small
    p means the difference is NOT a pure center shift -> slopes/shapes differ.
    Returns the p_value."""
    _, sigma_ref = fit_erf(rates_ref)
    x = to_lograte(rates_test)
    mu = x.mean()                      # MLE center with slope fixed to sigma_ref
    stat_obs = cramervonmises(x, erf_cdf, args=(mu, sigma_ref)).statistic
    n = len(x)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.normal(mu, sigma_ref, size=n)
        boot[b] = cramervonmises(sample, erf_cdf, args=(sample.mean(), sigma_ref)).statistic
    return (np.sum(boot >= stat_obs) + 1) / (n_boot + 1)

# statistics
############

rng = np.random.default_rng(0)
xgrid = np.linspace(np.log10(rate_lims[0]), np.log10(rate_lims[1]), 300)
stats = {}   # (drug, cond_a, cond_b) -> dict of results

print(f"\n=== rate-distribution statistics ({spn_titles[cell_type]}) ===")
for drug in drugs:
    print(f"\n{drug}")
    for cond_a, cond_b in comparisons:
        ra, rb = get_rates(drug, cond_a, cell_type), get_rates(drug, cond_b, cell_type)
        if len(ra) == 0 or len(rb) == 0:
            continue
        mu_a, sigma_a = fit_erf(ra)
        mu_b, sigma_b = fit_erf(rb)
        p_center = center_test(ra, rb, rng)
        p_slope = slope_test(ra, rb, rng)
        p_overall = cramervonmises_2samp(to_lograte(ra), to_lograte(rb)).pvalue
        stats[(drug, cond_a, cond_b)] = dict(
            center_a=mu_a, center_b=mu_b, slope_a=1.0 / sigma_a, slope_b=1.0 / sigma_b,
            p_center=p_center, p_slope=p_slope, p_overall=p_overall)
        print(f"  {cond_a} vs {cond_b} (n={len(ra)},{len(rb)}): "
              f"centers {mu_a:.2f} vs {mu_b:.2f} (p={p_center:.4f}), "
              f"slopes {1.0/sigma_a:.2f} vs {1.0/sigma_b:.2f} (p={p_slope:.4f}), "
              f"overall CvM p={p_overall:.4f}")

# plotting
##########

matplotlib.use("TkAgg")
plt.rcParams["font.size"] = base_fontsize
plt.rcParams["lines.markersize"] = 12.0
plt.rcParams["lines.linewidth"] = 2.0
import seaborn as sb
sb.set_palette("colorblind")

# fixed condition colors (Control=blue, Amph=orange, A+Drug=green)
palette = sb.color_palette("colorblind")
cond_colors = {"Control": palette[0], "Amphetamine": palette[1], "A + Drug": palette[2]}

short = {"Control": "C", "Amphetamine": "A", "A + Drug": "A+D"}

def sig(p):
    return "*" if p < 0.05 else ""

# figure layout: one panel per drug for the chosen cell type, sized to the
# panel-B white space (points -> inches)
n = len(drugs)
ncols = min(3, n)
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=(fig_width_pt / 72.0, fig_height_pt / 72.0),
                         sharex=True, sharey=True, squeeze=False)
# pin the outer axes spines to panel C's axes box (as figure fractions)
fig.subplots_adjust(left=left_spine_pt / fig_width_pt, right=right_spine_pt / fig_width_pt,
                    top=margin_top, bottom=margin_bottom, wspace=subplot_wspace)
axes = axes.ravel()

for k, drug in enumerate(drugs):
    ax = axes[k]
    for cond in conditions:
        rates = get_rates(drug, cond, cell_type)
        if len(rates) == 0:
            continue
        # empirical CDF
        xs, ys = ecdf(rates)
        ax.plot(xs, ys, color=cond_colors[cond], label=cond, drawstyle="steps-post", lw=2.0)
        # fitted ERF (Gaussian-CDF) overlay
        mu, sigma = fit_erf(rates)
        ax.plot(10.0 ** xgrid, erf_cdf(xgrid, mu, sigma), color=cond_colors[cond],
                ls="--", lw=1.3, alpha=0.8)

    # annotate the pairwise test results
    lines = []
    for cond_a, cond_b in comparisons:
        s = stats.get((drug, cond_a, cond_b))
        if s is None:
            continue
        tag = f"{short[cond_a]} vs {short[cond_b]}"
        lines.append(f"{tag} centers: {s['center_a']:.2f} vs {s['center_b']:.2f} "
                     f"{sig(s['p_center'])}")
        lines.append(f"{tag} slopes: {s['slope_a']:.2f} vs {s['slope_b']:.2f} "
                     f"{sig(s['p_slope'])}")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=annot_fontsize, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    ax.set_xscale("log")
    # extend the upper limit ~0.35 decade past the winsorization cap so the
    # 10^-3 tick is inset from the right spine (which sits at the page edge)
    ax.set_xlim(rate_lims[0], rate_lims[1] * 10 ** 0.35)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(drug)
    if k % ncols == 0:
        ax.set_ylabel("cumulative prob.")
    if k >= (nrows - 1) * ncols:
        ax.set_xlabel("mean OASIS rate (a.u.)")
    # single compact, frameless legend tucked into the empty lower-right corner
    # of the first panel; avoids colliding with the stats annotation
    if k == 0:
        ax.legend(loc="lower right", fontsize=annot_fontsize + 1.0, frameon=False,
                  handlelength=1.0, labelspacing=0.3, borderaxespad=0.3)

# hide unused axes
for k in range(n, len(axes)):
    axes[k].set_visible(False)

if show_suptitle:
    fig.suptitle(f"SPN rate distributions - {spn_titles[cell_type]} "
                 f"(dashed = ERF fit)", y=1.0)

# saving/plotting
fig.canvas.draw()
plt.savefig(f"{path}/spn_rate_distributions_{cell_type}.svg", format="svg")
plt.show()
