from scipy.special import rel_entr
import numpy as np
from scipy.stats import cauchy, norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import pickle
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


def kld_normal_cauchy(delta: float, mu: float, sd: float, x: np.ndarray):
    pdf_g = norm.pdf(x, loc=mu, scale=sd)
    pdf_c = cauchy.pdf(x, loc=mu, scale=delta)
    pdf_g /= np.sum(pdf_g)
    pdf_c /= np.sum(pdf_c)
    return np.sum(rel_entr(pdf_c, pdf_g))


def sample_sd_cauchy(delta: float, mu: float, sd: float, n_samples: int, lb: float, ub: float):
    samples_c = lorentzian(n_samples, eta=mu, delta=delta, lb=lb, ub=ub)
    samples_g = gaussian(n_samples, mu=mu, sd=sd, lb=lb, ub=ub)
    return (np.std(samples_c) - np.std(samples_g))**2


def repeated_sampling(delta, n: int, *args):
    return np.mean([sample_sd_cauchy(delta, *args) for _ in range(n)])


# parameters
mu = -40.0
lb = -60
ub = -20.0
n = 10000
n_reps = 5
sds = np.linspace(1.0, 6.0, 50)
bounds = [np.min(sds)*0.05, np.max(sds)+2.0]

# fit deltas to SDs
deltas = []
deltas_var = []
for sd in sds:
    deltas_tmp = []
    for _ in range(n_reps):
        delta = minimize_scalar(repeated_sampling, bounds=bounds, args=(1, mu, sd, n, lb, ub), method='bounded',
                                options={'maxiter': 1000, 'disp': True})
        deltas_tmp.append(delta.x)
    deltas.append(np.mean(deltas_tmp))
    deltas_var.append(np.var(deltas_tmp))

# calculate errors for two target SDs
sd_examples = [2.5, 5.0]
errors = {key: [] for key in sd_examples}
for sd in sd_examples:
    deltas1 = np.linspace(0.01, sd, num=100)
    errors_tmp = []
    for _ in range(n_reps):
        errors_tmp.append(np.asarray([repeated_sampling(delta, 1, mu, sd, n, lb, ub) for delta in deltas]))
    errors[sd] = (np.mean(errors_tmp, axis=0), np.var(errors_tmp, axis=1))

# save results
pickle.dump({"norm": sds, "lorentz": deltas, "norm_examples": sd_examples, "errors": errors, "var": deltas_var},
            open("results/norm_lorentz_fit.pkl", "wb"))

############
# plotting #
############

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# create figure with grid
fig = plt.figure(figsize=(12, 5))
grid = fig.add_gridspec(nrows=2, ncols=2)

# plot fitted deltas against sds
ax = fig.add_subplot(grid[0, 0])
ax.plot(sds, deltas, color="black")
for sd in sd_examples:
    idx = np.argmin(np.abs(sds - sd))
    delta = deltas[idx]
    ax.hlines(y=delta, xmin=np.min(sds), xmax=sd, color="red", linestyles="--")
    ax.vlines(x=sd, ymin=np.min(deltas), ymax=delta, color="red", linestyles="--")
ax.set_ylabel(r"$\Delta_v$ (mV)")
ax.set_xlabel(r"$\sigma_v$ (mV)")
ax.set_title(r"Fitted widths $\Delta_v$")

# plot error landscape for the two example SDs
ax = fig.add_subplot(grid[0, 1])
for sd in sd_examples:
    ax.plot(np.round(deltas, decimals=1), errors[sd][0], label=fr"$\sigma_v = {sd}$ mV")
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
    ax.hist(samples_l, bins=100, density=True, color="blue", label="Lorentzian")
    ax.hist(samples_g, bins=100, density=True, color="orange", label="Gaussian", alpha=0.6)
    ax.set_title(fr"$\sigma_v = {sd}$ mV, $Delta_v$ = {np.round(delta, decimals=1)}")
    ax.set_ylabel(r"$p$")
    ax.set_xlabel(r"$v_{\theta}$")
    ax.set_xlim([-60.0, -20.0])
    ax.legend()

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.show()
