from scipy.io import loadmat
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
import seaborn as sns

# preparations
##############

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6.9, 4.5)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.6
plt.rcParams['axes.titlepad'] = 3.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 15
cmap1 = sns.color_palette("plasma", as_cmap=False, n_colors=4)
cmap2 = sns.color_palette("plasma", as_cmap=True, n_colors=20)

# load matlab data
matlab_path = "/home/rgast/MatlabProjects/QIFSimulations/matlab_scripts/neco"
data = loadmat(f"{matlab_path}/neco_fig6_data.mat")

Ns = [1000, 2000, 4000, 8000]
Ps = [0.01, 0.03, 0.1, 0.3, 1.0]

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=4, ncols=5, figure=fig)

# plot frequency differences
ax1 = fig.add_subplot(grid[:2, :2])
freq_diffs = data['freq_mac'] - data['freqs']
im1 = ax1.imshow(freq_diffs, cmap=cmap2, aspect='auto', origin='lower', vmin=-2.4, vmax=2.4)
ax1.set_title('A: Frequency difference')
ax1.set_xlabel(r'$p$')
ax1.set_ylabel(r'$N$')
ax1.set_yticklabels([""] + [f"{n}" for n in Ns])
ax1.set_xticklabels(["", "0.01", "0.1", "1.0"])

# plot amplitude differences
ax2 = fig.add_subplot(grid[2:, :2])
amp_diffs = data['amp_mac'] - data['amps']
im2 = ax2.imshow(amp_diffs, cmap=cmap2, aspect='auto', origin='lower', vmin=-0.7, vmax=0.7)
ax2.set_title('B: Amplitude difference')
ax2.set_xlabel(r'$p$')
ax2.set_ylabel(r'$N$')
ax2.set_yticklabels([""] + [f"{n}" for n in Ns])
ax2.set_xticklabels(["", "0.01", "0.1", "1.0"])

# plot firing rates
row_idx = [0, 0, 0, 2]
col_idx = [0, 2, 4, 2]
axes = ["C", "D", "E", "F"]
t_start, t_end = 2000, 4200
time = np.linspace(0, (t_end-t_start)/10, t_end-t_start)
for i, (r, c, t) in enumerate(zip(row_idx, col_idx, axes)):

    n, p = Ns[r], Ps[c]
    ax = fig.add_subplot(grid[i, 2:])
    ax.plot(time, data['r_mac_rec_av'].squeeze()[t_start:t_end], c=to_hex(cmap1[-1]))
    ax.plot(time, data['rates'][r][c].squeeze()[t_start:t_end], c=(0, 0, 0, 1))
    ax.set_title(fr'{t}: $N = {n}$, $p = {p}$')
    ax.set_ylabel(r'$r$')
    ax.set_yticks([0, 1, 2])

ax.set_xlabel('time')
plt.legend(['mean-field model', 'QIF network'])

# plot color bars
fig.colorbar(im1, ax=ax1, shrink=0.8)
fig.colorbar(im2, ax=ax2, shrink=0.8)

# final touches
###############

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.05, hspace=0., wspace=0.)

# saving
fig.canvas.draw()
plt.savefig(f'neco_finite_size.svg')
plt.show()
