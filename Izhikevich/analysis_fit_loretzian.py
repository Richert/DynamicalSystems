from scipy.special import rel_entr
import numpy as np
from scipy.stats import cauchy, norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
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
sd = 6.0
lb = -70
ub = -10
n = 10000
bounds = [0.01, 10.0]
delta = minimize_scalar(sample_sd_cauchy, bounds=bounds, args=(mu, sd, n, lb, ub), method='bounded',
                        options={'maxiter': 1000, 'disp': True})
print(fr'Optimal Cauchy scale $\Delta$ for Gaussian mean $\mu = {mu}$ and $\sigma = {sd}$: {delta.x}')

# plot KLD for multiple delta values
deltas = np.linspace(0.01, sd, num=100)
errors = np.asarray([sample_sd_cauchy(delta, mu, sd, n, lb, ub) for delta in deltas])
fig, axes = plt.subplots(nrows=2)
axes[0].plot(deltas, errors)
axes[0].set_xlabel(r'$\Delta$')
axes[0].set_ylabel('Squared Error')
x = np.linspace(lb, ub, n)
pdf_g = norm.pdf(x, loc=mu, scale=sd)
pdf_c = cauchy.pdf(x, loc=mu, scale=delta.x)
axes[1].plot(x, pdf_g)
axes[1].plot(x, pdf_c)
plt.legend(["Gauss", "Cauchy"])
axes[1].set_ylabel("p(x)")
axes[1].set_xlabel("x")

n_samples = 10000
samples = lorentzian(n_samples, eta=mu, delta=delta.x, lb=lb, ub=ub)
axes[1].hist(samples, bins=100, density=True)
print(f"Sample SD: {np.std(samples)}")
plt.show()
