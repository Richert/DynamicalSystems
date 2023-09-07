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


mu = -40.0
lb = -70
ub = -10
n = 10000
sds = np.linspace(0.5, 6.0, 50)
bounds = [0.01, np.max(sds)]
deltas = []
for sd in sds:
    delta = minimize_scalar(sample_sd_cauchy, bounds=bounds, args=(mu, sd, n, lb, ub), method='bounded',
                            options={'maxiter': 1000, 'disp': True})
    # print(fr'Optimal Cauchy scale $\Delta$ for Gaussian mean $\mu = {mu}$ and $\sigma = {sd}$: {delta.x}')
    deltas.append(delta.x)

pickle.dump({"norm": sds, "lorentz": deltas}, open("results/norm_lorentz_fit.pkl", "wb"))

fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
axes[0].plot(deltas, sds)
axes[0].set_xlabel("Delta")
axes[1].set_ylabel("SD")
idx = 10
n_samples = 10000
samples_l = lorentzian(n_samples, eta=mu, delta=deltas[idx], lb=lb, ub=ub)
samples_g = gaussian(n_samples, mu=mu, sd=sds[idx], lb=lb, ub=ub)
axes[1].hist(samples_l, bins=100, density=True, color="blue", label="Lorentzian")
axes[1].hist(samples_g, bins=100, density=True, color="orange", label="Gaussian", alpha=0.6)
axes[1].set_title(f"SD = {sds[idx]}, Delta = {deltas[idx]}")
axes[1].set_ylabel(r"$p$")
axes[1].set_xlabel(r"$v_{\theta}$")
axes[1].legend()
plt.show()
