import numpy as np
import pickle
import matplotlib.pyplot as plt
from os import listdir
from pyrecu import sort_via_modules
from matplotlib import cm
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Roboto"
# plt.rc('text', usetex=True)
#plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14.0
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# get filenames
path = 'results'
n_files = 7

# get eta distribution
N = 1000
v_t = -40.0
Delta = 2.0
theta_dist = v_t + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# collect data
example_idx = 5
ps, modules_het, modules_hom, standard_errors, module_ex = [], [], [], [], []
for n in range(n_files):
    data_het = pickle.load(open(f"{path}/rnn_het_{n}.p", "rb"))
    data_hom = pickle.load(open(f"{path}/rnn_hom_{n}.p", "rb"))
    ps.append(data_het['p'])
    mods = [len(m) for m in data_het['modules']]
    modules_het.append((np.mean(mods), np.std(mods)))
    mods = [len(m) for m in data_hom['modules']]
    modules_hom.append((np.mean(mods), np.std(mods)))
    means = [np.mean(theta_dist[data_het['W'][n][i, :] > 0]) for i in range(N) for n in range(len(data_het['W']))]
    standard_errors.append(np.std(means))
    module_ex.append((data_het['modules'][example_idx], data_het['adjacency'][example_idx],
                      data_het['nodes'][example_idx], data_het['v'][example_idx]))

# get example modules
idx = np.argmin(np.abs([p-0.16 for p in ps]))
p = ps[idx]
mods, A, nodes, signal = module_ex[idx]
C = sort_via_modules(A, mods)

# plot data
fig, axes = plt.subplots(ncols=3, nrows=2)

ax1 = axes[0, 2]
ax1.bar(ps, standard_errors, color='grey')
ax1.set_xlabel('p')
ax1.set_ylabel('SEM')

ax2 = axes[1, 0]
x1 = np.arange(-0.2, len(ps)-0.2, step=1)
x2 = np.arange(0.2, len(ps), step=1)
ax2.bar(x1, [m[0] for m in modules_het], yerr=[m[1] for m in modules_het], color='grey')
ax2.bar(x2, [m[0] for m in modules_hom], yerr=[m[1] for m in modules_hom], color='blue')
ax2.set_xticks(np.arange(0, len(ps), step=2), labels=np.asarray(ps)[::2])
ax2.set_xlabel('p')
ax2.set_ylabel('# communities')
ax3 = axes[1, 1]
C1 = C[:, :]
C1[C1 > 0] = 1.0
ax3.imshow(C1, cmap='magma', interpolation='none')
ax3.set_xlabel('neuron id')
ax3.set_ylabel('neuron id')
ax4 = axes[1, 2]
cmap = cm.get_cmap('tab10')
x, y = 0, 0
for key, (indices, _) in mods.items():
    mean_signal = gaussian_filter1d(np.mean(signal[9000:9500, nodes[indices]], axis=1), sigma=10)
    ax4.plot(mean_signal, c=cmap(key-1))
    inc = len(indices)
    rect = Rectangle([x, y], inc, inc, edgecolor=cmap(key-1), facecolor='none')
    ax3.add_patch(rect)
    x += inc
    y += inc

sig = gaussian_filter1d(np.mean(signal[9000:9500, :], axis=1), sigma=10)
ax4.plot(sig, c='black')
ax4.set_xlabel('time (ms)')
ax4.set_ylabel('v (mV)')
plt.tight_layout()

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/ik_manifolds.svg')
plt.show()
