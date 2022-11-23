import numpy as np
import pickle
from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyrecu import sort_via_modules
from matplotlib import cm
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 5)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# get filenames
path = 'results'
n_files = 20

# get eta distribution
N = 1000
v_t = -40.0
deltas = np.arange(0.2, 4.1, 0.2)


# collect data
example_idx = 0
modules, standard_errors, module_ex = [], [], []
for n in range(n_files):
    data = pickle.load(open(f"{path}/rnn_{n}.p", "rb"))
    mods = [len(m) for m in data['modules']]
    modules.append((np.mean(mods), np.std(mods)))
    theta_dist = v_t + deltas[n]*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))
    means = [np.mean(theta_dist[data['W'][n][i, :] > 0]) for i in range(N) for n in range(len(data['W']))]
    standard_errors.append(np.std(means))
    module_ex.append((data['modules'][example_idx], data['adjacency'][example_idx],
                      data['nodes'][example_idx], data['v'][example_idx]))

# get example modules
d = 1.0
idx = np.argmin(np.abs(deltas - d))
mods, A, nodes, signal = module_ex[idx]
C = sort_via_modules(A, mods)

# plot data
# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

ax0 = fig.add_subplot(grid[:2, 0])
ax0.set_title("(A) Sparse synapses sample from \n heterogeneous network units")

ax1 = fig.add_subplot(grid[2, 0])
ax1.plot(deltas, [m[0] for m in modules])
ax1.fill_between(deltas, [m[0] - m[1] for m in modules], [m[0] + m[1] for m in modules], alpha=0.3)
ax1.set_xticks(deltas[::6])
ax1.set_xlabel(r'$\Delta_v$')
ax1.set_ylabel('n')
ax1.set_title("(C) Number of communities \n" + r"varies with $\Delta_v$")

ax2 = fig.add_subplot(grid[:2, 1])
C1 = C[:, :]
C1[C1 > 0] = 1.0
ax2.imshow(C1, cmap='magma', interpolation='none')
ax2.set_xlabel('neuron id')
ax2.set_ylabel('neuron id')
ax2.set_xticks(np.arange(0, 900, 400))
ax2.set_title(rf"(B) Community structure ($\Delta_v = {d}$)")

ax3 = fig.add_subplot(grid[2, 1])
cmap = cm.get_cmap('tab10')
x, y = 0, 0
for key, (indices, _) in mods.items():
    mean_signal = gaussian_filter1d(np.mean(signal[9000:9500, nodes[indices]], axis=1), sigma=10)
    ax3.plot(mean_signal, c=cmap(key-1))
    inc = len(indices)
    rect = Rectangle([x, y], inc, inc, edgecolor=cmap(key-1), facecolor='none')
    ax2.add_patch(rect)
    x += inc
    y += inc

sig = gaussian_filter1d(np.mean(signal[9000:9500, :], axis=1), sigma=10)
ax3.plot(sig, c='black')
ax3.set_xlabel('time (ms)')
ax3.set_ylabel('v (mV)')
ax3.set_title("(D) Communities exhibit distinct \n mean-field dynamics")

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/ik_manifolds2.svg')
plt.show()
