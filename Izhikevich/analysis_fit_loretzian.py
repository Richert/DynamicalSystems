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


def kld_normal_cauchy(delta: float, mu: float, sd: float, x: np.ndarray):
    pdf_g = norm.pdf(x, loc=mu, scale=sd)
    pdf_c = cauchy.pdf(x, loc=mu, scale=delta)
    pdf_g /= np.sum(pdf_g)
    pdf_c /= np.sum(pdf_c)
    return np.sum(rel_entr(pdf_c, pdf_g))


mu = -40.0
sd = 2.0
bounds = [0.01, 10.0]
x = mu + np.linspace(-50.0, 50.0, num=10000)
delta = minimize_scalar(kld_normal_cauchy, bounds=bounds, args=(mu, sd, x), method='bounded',
                        options={'maxiter': 1000, 'disp': True})
print(fr'Optimal Cauchy scale $\Delta$ for Gaussian mean $\mu = {mu}$ and $\sigma = {sd}$: {delta.x}')

# plot KLD for multiple delta values
deltas = np.linspace(0.1, 10.0, num=1000)
klds = np.asarray([kld_normal_cauchy(delta, mu, sd, x) for delta in deltas])
fig, axes = plt.subplots(nrows=2)
axes[0].plot(deltas, klds)
axes[0].set_xlabel(r'$\Delta$')
axes[0].set_ylabel('KLD')
pdf_g = norm.pdf(x, loc=mu, scale=sd)
pdf_c = cauchy.pdf(x, loc=mu, scale=delta.x)
axes[1].plot(x, pdf_g)
axes[1].plot(x, pdf_c)
plt.legend(["Gauss", "Cauchy"])
axes[1].set_ylabel("p(x)")
axes[1].set_xlabel("x")

n_samples = 200
samples = lorentzian(n_samples, eta=mu, delta=delta.x, lb=-60, ub=-10)
axes[1].hist(samples, bins=20, density=True)
print(f"Sample SD: {np.std(samples)}")
plt.show()
