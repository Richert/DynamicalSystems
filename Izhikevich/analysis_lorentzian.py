import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib import gridspec
from scipy.stats import cauchy

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.handlelength'] = 1.0
markersize = 6

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# plot truncated lorentzian distributions
n = 10000
v_t = -50.0
lb = -70.0
ub = -30.0
deltas = np.asarray([2.0, 4.0, 8.0])
potentials = np.linspace(-100, 0, n)
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
ax.set_xlabel(r'$v_{\theta}$')
ax.set_ylabel(r'$p(v_{\theta})$')
plt.legend(lines, [fr'$\Delta_v = {d}$' for d in deltas])

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/lorentzian.pdf')
plt.show()
