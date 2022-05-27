import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib import gridspec
from scipy.stats import cauchy
from scipy.signal import welch
import pickle

# load simulation data
data = pickle.load(open(f"results/rs_lorentzian.p", "rb"))['results']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.handlelength'] = 1.0
markersize = 6


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def get_potential(v_t: float, v_r: float, I: float, k: float):
    mu = (v_r+v_t)**2/4 - v_r*v_t - I/k
    if mu < 0:
        return v_t
    return (v_r + v_t)/2 + np.sqrt(mu)


def get_potential_avg(n: int, v_t: float, delta: float, lb: float, ub: float, v_r: float, I: float, k: float):
    thresholds = lorentzian(n, v_t, delta, lb, ub)
    rhos = np.asarray([cauchy.pdf(v, loc=v_t, scale=delta) for v in thresholds if lb < v < ub])
    rhos /= np.sum(rhos)
    ps = np.asarray([get_potential(v, v_r, I, k) for v in thresholds if lb < v < ub])
    return np.dot(rhos, ps)


# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=3, ncols=1, figure=fig)

# plot truncated lorentzian distributions
n = 10000
v_t = -40.0
lb = -60.0
ub = -20.0
deltas = np.asarray([1.0, 2.0, 3.0, 4.0])
potentials = np.linspace(-80, 0, n)
cmap = plt.get_cmap('copper', lut=len(deltas))
ax = fig.add_subplot(grid[0, 0])
lines = []
for i, delta in enumerate(deltas):
    pdf = cauchy.pdf(potentials, loc=v_t, scale=delta)
    idx = np.asarray([lb < p < ub for p in potentials])
    pdf_truncated = pdf[idx]
    c = to_hex(cmap(i, alpha=1.0))
    ax.plot(potentials, pdf, linestyle='--', color=c)
    line = ax.plot(potentials[idx], pdf_truncated, color=c)
    lines.append(line[0])
ax.set_xlim([-80.0, 0.0])
ax.set_xlabel(r'$v_{\theta}$')
ax.set_ylabel(r'$p(v_{\theta})$')
plt.legend(lines, [fr'$\Delta_v = {d}$' for d in deltas])

# plot mean and variance of difference between MF and SNN
n = 100
deltas2 = np.linspace(0.1, 10.0, num=n)
diff_mean, snn_var = [], []
for i, d in enumerate(deltas2):
    fre = data['fre'][i]
    snn = data['snn'][i]
    diff = fre['s'][:, 0] - snn['s'][:, 0]
    # if i % 10 == 0:
    #     fig, ax = plt.subplots(figsize=(8, 3))
    #     ax.plot(fre.index, snn['v'])
    #     ax.plot(fre['v'])
    #     plt.show()
    diff_mean.append(np.mean(diff))
    snn_var.append(np.var(snn['s'][:, 0]))
ax = fig.add_subplot(grid[1, 0])
ax.plot(deltas2, diff_mean, color='blue')
ax2 = ax.twinx()
ax2.plot(deltas2, snn_var, color='orange')
ax.set_xlabel(r'$\Delta_v$')
ax.set_xlim([0, 5.0])
# ax.set_ylabel(r'$\text{mean}(v_{mf}(t) - v_{snn}(t))$')
# ax2.set_ylabel(r'$\text{var}(v_{mf}(t) - v_{snn}(t))$')

# plot different power spectra for a bunch of deltas
ax = fig.add_subplot(grid[2, 0])
for delta in deltas:
    idx = np.argmin(np.abs(deltas2-delta))
    snn = data['snn'][idx]
    freqs, pow = welch(snn['s'][:, 0], fs=1e4, nperseg=1024)
    ax.semilogy(freqs, pow)
ax.set_xlabel(r'frequency ($Hz$)')
ax.set_ylabel(r'PSD ($r^2/Hz$)')

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/lorentzian.pdf')
plt.show()
