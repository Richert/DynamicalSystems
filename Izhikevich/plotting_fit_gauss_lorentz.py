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
sd_examples = results["norm_examples"]
errors = results["errors"]
deltas_v = np.asarray(results["var"])

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
ax.plot(sds, deltas, color="black")
# ax.fill_between(sds, deltas-deltas_v, deltas+deltas_v, alpha=0.5, color="black")
colors = ["royalblue", "darkorange"]
for sd, c in zip(sd_examples, colors):
    idx = np.argmin(np.abs(sds - sd))
    delta = deltas[idx]
    ax.hlines(y=delta, xmin=np.min(sds), xmax=sd, color=c, linestyles="--")
    ax.vlines(x=sd, ymin=np.min(deltas), ymax=delta, color=c, linestyles="--")
ax.set_ylabel(r"$\Delta_v$ (mV)")
ax.set_xlabel(r"$\sigma_v$ (mV)")
ax.set_title(r"Fitted widths $\Delta_v$")

# plot error landscape for the two example SDs
ax = fig.add_subplot(grid[0, 1])
colors = ["royalblue", "darkorange"]
for sd, c in zip(sd_examples, colors):
    ax.plot(np.round(deltas, decimals=1), errors[sd][0], "x", label=fr"$\sigma_v = {sd}$ mV", color=c)
    # for x, y, y_v in zip(deltas, errors[sd][0], errors[sd][1]):
    #     ax.vlines(x=x, ymin=y-y_v, ymax=y+y_v, color=c)
ax.set_xlabel(r"$\Delta_v$ (mV)")
ax.set_ylabel("error")
ax.set_title("Squared error between sample SDs")
ax.legend()

# plot the histograms for the two examples
n_samples = 10000
for i, sd in enumerate(sd_examples):
    ax = fig.add_subplot(grid[1, i])
    idx = np.argmin(np.abs(sds - sd))
    delta = deltas[idx]
    samples_l = lorentzian(n_samples, eta=mu, delta=delta, lb=lb, ub=ub)
    samples_g = gaussian(n_samples, mu=mu, sd=sd, lb=lb, ub=ub)
    ax.hist(samples_l, bins=100, density=True, color="dimgrey", label="Lorentzian")
    ax.hist(samples_g, bins=100, density=True, color="firebrick", label="Gaussian", alpha=0.6)
    ax.set_title(fr"$\sigma_v = {sd}$ mV, $Delta_v$ = {np.round(delta, decimals=1)}")
    ax.set_ylabel(r"$p$")
    ax.set_xlabel(r"$v_{\theta}$")
    ax.set_xlim([-60.0, -20.0])
    ax.legend()

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/lorentz_gauss_fitting.svg')
plt.show()
