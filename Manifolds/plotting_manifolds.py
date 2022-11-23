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
#plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
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
example_idx = 3
ps, modules_het, modules_hom, modules_med, standard_errors, module_ex = [], [], [], [], [], []
for n in range(1, n_files):
    data_het = pickle.load(open(f"{path}/rnn_het_{n}.p", "rb"))
    data_med = pickle.load(open(f"{path}/rnn_med_{n}.p", "rb"))
    data_hom = pickle.load(open(f"{path}/rnn_hom_{n}.p", "rb"))
    ps.append(data_het['p'])
    modules_het.append(data_het['dim'])
    modules_med.append(data_med['dim'])
    modules_hom.append(data_hom['dim'])
    means = [np.mean(theta_dist[data_hom['W'][n][i, :] > 0]) for i in range(N) for n in range(len(data_hom['W']))]
    standard_errors.append(np.std(means))
    module_ex.append((data_hom['modules'][example_idx], data_hom['adjacency'][example_idx],
                      data_hom['nodes'][example_idx], data_hom['s'][example_idx]))

# get example modules
p_t = 0.16
idx = np.argmin(np.abs([p-p_t for p in ps]))
p = ps[idx]
mods, A, nodes, signal = module_ex[idx]
C = sort_via_modules(A, mods)

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=5, ncols=2, figure=fig)

ax0 = fig.add_subplot(grid[:3, 0])
ax0.set_title("(A) Sparse synapses sample from \n heterogeneous network units")

# clustering
ax1 = fig.add_subplot(grid[:3, 1])
C1 = C[:, :]
C1[C1 > 0] = 1.0
ax1.imshow(C1, cmap='magma', interpolation='none')
ax1.set_xlabel('neuron id')
ax1.set_ylabel('neuron id')
ax1.set_title("(C) Community structure of \n" + rf"network dynamics ($\Delta_v = 0.2$, $p = {p_t}$)")

# number of communities
ax2 = fig.add_subplot(grid[3:, 0])
x1 = np.arange(-0.3, len(ps)-0.3, step=1)
x2 = np.arange(0.0, len(ps), step=1)
x3 = np.arange(0.3, len(ps)+0.3, step=1)
cmap = plt.get_cmap("tab10")
ax2.bar(x1, [m[0] for m in modules_hom], yerr=[m[1] for m in modules_hom], color=cmap(0), width=0.2,
        label=r"$\Delta_v = 0.2$")
ax2.bar(x2, [m[0] for m in modules_med], yerr=[m[1] for m in modules_hom], color=cmap(1), width=0.2,
        label=r"$\Delta_v = 1.0$")
ax2.bar(x3, [m[0] for m in modules_het], yerr=[m[1] for m in modules_hom], color=cmap(2), width=0.2,
        label=r"$\Delta_v = 2.0$")
plt.legend()
ax2.set_xticks(np.arange(0, len(ps), step=2), labels=np.asarray(ps)[::2])
ax2.set_xlabel(r'$p$')
ax2.set_ylabel(r'$n$')
ax2.set_title("(B) Dimensionality of network \n" + r"dynamics varies with $\Delta_v$")

# time series
ax3 = fig.add_subplot(grid[3:, 1])
cmap = cm.get_cmap('tab10')
x, y = 0, 0
for key, (indices, _) in mods.items():
    mean_signal = gaussian_filter1d(np.mean(signal[9000:, nodes[indices]], axis=1), sigma=10)
    ax3.plot(mean_signal, c=cmap(key-1))
    inc = len(indices)
    rect = Rectangle([x, y], inc, inc, edgecolor=cmap(key-1), facecolor='none')
    ax1.add_patch(rect)
    x += inc
    y += inc
sig = gaussian_filter1d(np.mean(signal[9000:, :], axis=1), sigma=10)
ax3.plot(sig, c='black')
ax3.set_xticks([0, 500, 1000], labels=[0, 1000, 2000])
ax3.set_xlabel('time (ms)')
ax3.set_ylabel(r'$s$')
ax3.set_title("(D) Communities exhibit distinct \n mean-field dynamics")
plt.tight_layout()

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/ik_manifolds.svg')
plt.show()
