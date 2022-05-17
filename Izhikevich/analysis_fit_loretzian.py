from scipy.special import rel_entr
import numpy as np
from scipy.stats import cauchy, norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


def kld_normal_cauchy(delta: float, mu: float, sd: float, x: np.ndarray):
    pdf_g = norm.pdf(x, loc=mu, scale=sd)
    pdf_c = cauchy.pdf(x, loc=mu, scale=delta)
    pdf_g /= np.sum(pdf_g)
    pdf_c /= np.sum(pdf_c)
    return np.sum(rel_entr(pdf_g, pdf_c))


mu = -40.0
sd = 3.0
bounds = [0.01, 10.0]
x = mu + np.linspace(-100.0, 100.0, num=10000)
delta = minimize_scalar(kld_normal_cauchy, bounds=bounds, args=(mu, sd, x), method='bounded',
                        options={'maxiter': 1000, 'disp': True})
print(fr'Optimal Cauchy scale $\Delta$ for Gaussian mean $\mu = {mu}$ and $\sigma = {sd}$: {delta.x}')

# plot KLD for multiple delta values
deltas = np.linspace(0.1, 10.0, num=1000)
klds = np.asarray([kld_normal_cauchy(delta, mu, sd, x) for delta in deltas])
plt.plot(deltas, klds)
plt.xlabel(r'$\Delta$')
plt.ylabel('KLD')
plt.show()
