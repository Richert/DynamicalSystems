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
plt.rcParams['figure.figsize'] = (6, 6)
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
        diff = 1e3*(fre['r'][:, 0] - snn['r'][:, 0])
        diffs.append(np.mean(diff))
        variances.append(np.var(diff))
    return diffs, variances


def plot_psd(data: dict, vals: np.ndarray, targets: np.ndarray, ax: plt.Axes, fs: float = 1e4, nperseg: int = 2048):
    cmap = plt.get_cmap('copper', lut=len(targets))
    for i, val in enumerate(targets):
        idx = np.argmin(np.abs(vals - val))
        snn = data['snn'][idx]
        freqs, pow = welch(1e3*snn['r'][:, 0], fs=fs, nperseg=nperseg)
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
lb_samples = [(-70.0, -10.0), (-60.0, -20.0)]
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
plt.title('(A) Truncated Lorentzian PDFs')

# plot mean and variance of difference between MF and SNN
diff_delta, var_delta = get_diff_and_var(data_delta)
diff_lb, var_lb = get_diff_and_var(data_lbs)
ax = fig.add_subplot(grid[1, 0])
ax.plot(deltas, diff_delta, color='blue')
ax.set_ylabel(r'mean of $D_r$ ($Hz$)', color='blue')
fig.canvas.draw()
ax.set_yticklabels(ax.get_yticklabels(), color='blue')
ax2 = ax.twinx()
ax2.plot(deltas, var_delta, color='orange')
fig.canvas.draw()
ax2.set_yticklabels(ax2.get_yticklabels(), color='orange')
ax.set_xlabel(r'$\Delta_v$')
ax.set_xlim([0.0, np.max(deltas)])
ax.set_xticks([0, 2, 4])
ax.set_yticks([-4, -2, 0])
plt.title('(B)')
ax = fig.add_subplot(grid[1, 1])
ax.plot(v_t-lbs, diff_lb, color='blue')
fig.canvas.draw()
ax.set_yticklabels(ax.get_yticklabels(), color='blue')
ax2 = ax.twinx()
ax2.plot(v_t-lbs, var_lb, color='orange')
ax2.set_ylabel(r'var of $D_r$ ($Hz$)', color='orange')
fig.canvas.draw()
ax2.set_yticklabels(ax2.get_yticklabels(), color='orange')
ax.set_xlabel(r'$\phi$')
ax.set_xlim([np.max(v_t-lbs), np.min(v_t-lbs)])
plt.title('(C)')

# plot different power spectra for a bunch of examples
delta_samples = np.asarray([0.5, 1.0, 2.0, 4.0])
lb_samples = np.asarray([-80.0, -70.0, -50.0, -45.0])
nperseg = 4096
ax = fig.add_subplot(grid[2, 0])
ax = plot_psd(data_delta, deltas, delta_samples, ax, nperseg=nperseg)
ax.set_xlim([0.0, 300.0])
plt.legend([fr'$\Delta_v = {d}$' for d in delta_samples], loc=4)
plt.title('(D)')
ax2 = fig.add_subplot(grid[2, 1])
ax2 = plot_psd(data_lbs, lbs, lb_samples, ax2, nperseg=nperseg)
ax2.set_xlim([0.0, 300.0])
ax2.set_ylim([1e-5, 5e-2])
ax2.set_ylabel('')
plt.legend([fr'$\phi = {v_t-v}$' for v in lb_samples], loc=4)
plt.title('(E)')
ax.set_ylim(ax2.get_ylim())

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, hspace=0., wspace=0.03)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/lorentzian.pdf')
plt.show()
