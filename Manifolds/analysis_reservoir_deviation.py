import numpy as np
import pickle
import matplotlib.pyplot as plt
from os import listdir
from pyrecu import sort_via_modules
from matplotlib import cm
from matplotlib.patches import Rectangle

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
path = 'C:\\Users\\rgf3807\\OneDrive - Northwestern University\\data'
fn = 'rnn_'
fns = [f for f in listdir(path) if fn in f]

# get eta distribution
N = 1000
eta = 45.0
Delta = 2.0
eta_dist = eta + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# collect data
example_idx = 0
ps, modules, standard_errors, module_ex = [], [], [], []
for f in fns:
    data = pickle.load(open(f"{path}/{f}", "rb"))
    ps.append(int(data['p']*N))
    mods = [len(m) for m in data['modules']]
    modules.append((np.mean(mods), np.std(mods)))
    means = [np.mean(eta_dist[data['W'][n][i, :] > 0]) for i in range(N) for n in range(len(data['W']))]
    standard_errors.append(np.std(means))
    module_ex.append((data['modules'][example_idx], data['adjacency'][example_idx], data['nodes'][example_idx],
                      data['v'][example_idx]))

# get example modules
idx = np.argmin(np.abs([p-0.05*N for p in ps]))
p = ps[idx]
mods, A, nodes, signal = module_ex[idx]
C = sort_via_modules(A, mods)

# plot data
fig, axes = plt.subplots(ncols=3, nrows=2)
ax1 = axes[0, 2]
x = np.argsort(ps)
ax1.bar(np.arange(0, len(x)), [standard_errors[idx] for idx in x], color='grey')
ax1.set_xticks(np.arange(0, len(x)), labels=[ps[idx] for idx in x])
ax1.set_xlabel('# incoming synapses')
ax1.set_ylabel('SEM')
ax2 = axes[1, 0]
ax2.bar(np.arange(0, len(x)), [m[0] for m in modules], yerr=[m[1] for m in modules], color='grey')
ax2.set_xticks(np.arange(0, len(x), step=2), labels=[ps[idx] for idx in x[::2]])
ax2.set_xlabel('# incoming synapses')
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
    mean_signal = np.mean(signal[9000:9800, nodes[indices]], axis=1)
    ax4.plot(mean_signal, c=cmap(key-1))
    inc = len(indices)
    rect = Rectangle([x, y], inc, inc, edgecolor=cmap(key-1), facecolor='none')
    ax3.add_patch(rect)
    x += inc
    y += inc

ax4.plot(np.mean(signal[9000:9800, :], axis=1), c='black')
ax4.set_xlabel('time (ms)')
ax4.set_ylabel('v (mV)')
plt.tight_layout()

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/ik_manifolds.svg')
plt.show()
