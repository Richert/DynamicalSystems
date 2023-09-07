import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import pickle
from scipy.stats import cauchy, norm


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


def get_fr(inp: np.ndarray, k: float, C: float, v_reset: float, v_spike: float, v_r: float, v_t: float):
    fr = np.zeros_like(inp)
    alpha = v_r+v_t
    mu = 4*(v_r*v_t + inp/k) - alpha**2
    idx = mu > 0
    mu_sqrt = np.sqrt(mu[idx])
    fr[idx] = k*mu_sqrt/(2*C*(np.arctan((2*v_spike-alpha)/mu_sqrt) - np.arctan((2*v_reset-alpha)/mu_sqrt)))
    return fr


# load data
mapping = pickle.load(open("results/norm_lorentz_fit.pkl", "rb"))

# parameters
inp = np.linspace(50.0, 100.0, num=100)
v_reset = -1000.0
v_spike = 1000.0
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
N = 1000
n_bins = 50

# find pair of Lorentzian width and Gaussian standard deviation
Delta = 0.6
idx = np.argmin(np.abs(np.asarray(mapping["lorentz"]) - Delta))
SD = mapping["norm"][idx]

# define lorentzian of etas
thetas_l = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2*v_t-v_r)
thetas_g = gaussian(N, mu=v_t, sd=SD, lb=v_r, ub=2*v_t-v_r)

# get firing rates of each neuron
fr_lorentz, fr_gauss = [], []
for v_l, v_g in zip(thetas_l, thetas_g):
    fr_lorentz.append(get_fr(inp, k, C, v_reset, v_spike, v_r, v_l))
    fr_gauss.append(get_fr(inp, k, C, v_reset, v_spike, v_r, v_g))
fr_lorentz = np.asarray(fr_lorentz)*1e3
fr_gauss = np.asarray(fr_gauss)*1e3

# get firing rate distributions for each input strength
rate_bins = np.linspace(np.min(fr_lorentz.flatten()), np.max(fr_lorentz.flatten()), n_bins+1)
p_lorentz, p_gauss = [], []
for idx in range(len(inp)):
    p_l, _ = np.histogram(fr_lorentz[:, idx], rate_bins, density=False)
    p_g, _ = np.histogram(fr_gauss[:, idx], rate_bins, density=False)
    p_lorentz.append(p_l / np.sum(p_l))
    p_gauss.append(p_g / np.sum(p_g))

# plotting
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

ax = axes[0]
im = ax.imshow(np.asarray(p_lorentz).T, cmap=plt.get_cmap('cividis'), aspect='auto', vmin=0.0, vmax=1.0, origin='lower')
ax.set_xlabel(r"$I$ (pA)")
ax.set_ylabel(r"$r_i$")
ax.set_title(f"Lorentzian - Delta = {Delta}")
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1]
im = ax.imshow(np.asarray(p_gauss).T, cmap=plt.get_cmap('cividis'), aspect='auto', vmin=0.0, vmax=1.0, origin='lower')
ax.set_xlabel(r"$I$ (pA)")
ax.set_ylabel(r"$r_i$")
ax.set_title(f"Gaussian - SD = {SD}")
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.show()
