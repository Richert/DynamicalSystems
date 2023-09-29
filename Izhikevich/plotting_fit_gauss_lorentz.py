import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import cauchy, norm
plt.rc('text', usetex=True)


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def gaussian(n, mu: float, sd: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = norm.rvs(loc=mu, scale=sd)
        while s <= lb or s >= ub:
            s = norm.rvs(loc=mu, scale=sd)
        samples[i] = s
    return samples


################
# preparations #
################

# load data
results = pickle.load(open("results/norm_lorentz_fit.pkl", "rb"))
sds = np.asarray(results["norm"])
deltas = np.asarray(results["lorentz"])
delta_examples = results["delta_examples"]
errors = results["errors"]
sd_var = np.asarray(results["var"])
sd_errors = np.asarray(results["norm_errors"])

# parameters
mu = -40.0
lb = -70
ub = -10
bounds = [0.01, np.max(sds)+2.0]

############
# plotting #
############

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 4

# create figure with grid
fig = plt.figure(figsize=(12, 5))
grid = fig.add_gridspec(nrows=2, ncols=2)

# plot fitted deltas against sds
ax = fig.add_subplot(grid[0, 0])
ax.plot(deltas, sds, color="black")
ax.fill_between(deltas, sds-np.sqrt(sd_var), sds+np.sqrt(sd_var), alpha=0.2, facecolor="black", edgecolor="none")
colors = ["royalblue", "darkorange"]
for delta, c in zip(delta_examples, colors):
    idx = np.argmin(np.abs(deltas - delta))
    sd = sds[idx]
    ax.hlines(y=sd, xmin=np.min(deltas), xmax=delta, color=c, linestyles="--")
    ax.vlines(x=delta, ymin=np.min(sds), ymax=sd, color=c, linestyles="--")
ax.set_xlabel(r"$\Delta_v$ (mV)")
ax.set_ylabel(r"$\sigma_v$ (mV)")
ax.set_title(r"Fitted SDs $\sigma_v$")

# plot error landscape for the two example SDs
ax = fig.add_subplot(grid[0, 1])
colors = ["royalblue", "darkorange"]
for delta, c in zip(delta_examples, colors):
    ax.plot(np.round(sd_errors, decimals=2), errors[delta][0], "x", label=fr"$\Delta_v = {delta}$ mV", color=c)
    # for x, y, y_v in zip(deltas, errors[sd][0], errors[sd][1]):
    #     ax.vlines(x=x, ymin=y-y_v, ymax=y+y_v, color=c)
ax.set_xlabel(r"$\sigma_v$ (mV)")
ax.set_ylabel("error")
ax.set_title("Squared error between sample SDs")
ax.legend()

# plot the histograms for the two examples
n_samples = 10000
for i, delta in enumerate(delta_examples):
    ax = fig.add_subplot(grid[1, i])
    idx = np.argmin(np.abs(deltas - delta))
    sd = sds[idx]
    samples_l = lorentzian(n_samples, eta=mu, delta=delta, lb=lb, ub=ub)
    samples_g = gaussian(n_samples, mu=mu, sd=sd, lb=lb, ub=ub)
    ax.hist(samples_l, bins=100, density=True, color="dimgrey", label="Lorentzian")
    ax.hist(samples_g, bins=100, density=True, color="firebrick", label="Gaussian", alpha=0.6)
    ax.set_title(fr"$\sigma_v = {np.round(sd, decimals=1)}$ mV, $\Delta_v$ = {np.round(delta, decimals=1)}")
    ax.set_ylabel(r"$p$")
    ax.set_xlabel(r"$v_{\theta}$")
    ax.set_xlim([-60.0, -20.0])
    # ax.set_yscale("log")
    ax.legend()

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/lorentz_gauss_fitting.svg')
plt.show()
