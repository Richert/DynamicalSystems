import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib import gridspec
from scipy.stats import cauchy
from scipy.signal import welch
import pickle

# load simulation data
data_delta = pickle.load(open(f"results/rs_lorentzian_deltas.p", "rb"))['results']
data_lbs = pickle.load(open(f"results/rs_lorentzian_lb.p", "rb"))['results']
deltas = pickle.load(open(f"results/rs_lorentzian_deltas.p", "rb"))['deltas']
lbs = pickle.load(open(f"results/rs_lorentzian_lb.p", "rb"))['lbs']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 5)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.handlelength'] = 1.0
markersize = 6


def get_diff_and_var(data: dict):
    diffs, variances = [], []
    for fre, snn in zip(data['fre'], data['snn']):
        diff = fre['r'][:, 0] - snn['r'][:, 0]
        diffs.append(np.mean(diff))
        variances.append(np.var(snn['r'][:, 0]))
    return diffs, variances


def plot_psd(data: dict, vals: np.ndarray, targets: np.ndarray, ax: plt.Axes):
    cmap = plt.get_cmap('copper', lut=len(targets))
    for i, val in enumerate(targets):
        idx = np.argmin(np.abs(vals - val))
        snn = data['snn'][idx]
        freqs, pow = welch(snn['r'][:, 0], fs=1e4, nperseg=4096)
        c = to_hex(cmap(i, alpha=1.0))
        ax.semilogy(freqs, pow, c=c)
    ax.set_xlabel(r'frequency ($Hz$)')
    ax.set_ylabel(r'PSD ($r^2/Hz$)')
    return ax


# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

# plot truncated lorentzian distributions
n = 10000
v_t = -40.0
lb = -60.0
ub = -20.0
delta_samples = np.asarray([1.0, 2.0, 3.0, 4.0])
lb_samples = [(-70.0, 10.0), (-60.0, -20.0)]
potentials = np.linspace(-80, 0, n)
cmap = plt.get_cmap('copper', lut=len(delta_samples))
ax = fig.add_subplot(grid[0, :])
lines = []
for i, delta in enumerate(delta_samples):
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
plt.legend(lines, [fr'$\Delta_v = {d}$' for d in delta_samples])
colors = ['grey', 'black']
for (x1, x2), c in zip(lb_samples, colors):
    ax.axvline(x=x1, color=c, linestyle='dotted')
    ax.axvline(x=x2, color=c, linestyle='dotted')

# plot mean and variance of difference between MF and SNN
diff_delta, var_delta = get_diff_and_var(data_delta)
diff_lb, var_lb = get_diff_and_var(data_lbs)
ax = fig.add_subplot(grid[1, 0])
ax.plot(deltas, diff_delta, color='blue')
ax.set_ylabel(r'$mean(r(t) - \langle r_i(t) \rangle_i)$')
ax2 = ax.twinx()
ax2.plot(deltas, var_delta, color='orange')
ax2.set_ylabel(r'$var(\langle r_i(t) \rangle_i)$')
ax.set_xlabel(r'$\Delta_v$')
ax.set_xlim([np.min(deltas), np.max(deltas)])
ax = fig.add_subplot(grid[1, 1])
ax.plot(lbs, diff_lb, color='blue')
ax.set_ylabel(r'$mean(r(t) - \langle r_i(t) \rangle_i)$')
ax2 = ax.twinx()
ax2.plot(lbs, var_lb, color='orange')
ax2.set_ylabel(r'$var(\langle r_i(t) \rangle_i)$')
ax.set_xlabel(r'$v_0$')
ax.set_xlim([np.min(lbs), np.max(lbs)])

# plot different power spectra for a bunch of examples
delta_samples = np.asarray([0.5, 1.0, 2.0, 4.0])
lb_samples = np.asarray([-80.0, -70.0, -60.0, -50.0])
ax = fig.add_subplot(grid[2, 0])
ax = plot_psd(data_delta, deltas, delta_samples, ax)
ax.set_xlim([0.0, 200.0])
ax.set_ylim([1e-11, 1e-7])
plt.legend([fr'$\Delta_v = {d}$' for d in delta_samples], loc=4)
ax = fig.add_subplot(grid[2, 1])
ax = plot_psd(data_lbs, lbs, lb_samples, ax)
ax.set_xlim([0.0, 200.0])
ax.set_ylim([1e-10, 1e-7])
plt.legend([fr'$v_0 = {v}$' for v in lb_samples], loc=4)

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/lorentzian.pdf')
plt.show()
