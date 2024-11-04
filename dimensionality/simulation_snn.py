from rectipy import Network, random_connectivity
from pyrates import OperatorTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from custom_functions import *

# define parameters
###################

# meta parameters
device = "cuda:0"
theta_dist = "lorentzian"

# general model parameters
N = 1000
p = 0.2
E_e = 0.0
E_i = -60.0
v_spike = 1000.0
v_reset = -1000.0

# get sweep condition
g = 5.0 #float(sys.argv[-1])
Delta = 0.5 #float(sys.argv[-2])

# input parameters
dt = 1e-2
dts = 1e-1
dur = 300.0
amp = 35.0
cutoff = 500.0
T = 1000.0 + cutoff
start = 500.0 + cutoff

# model parameters
C = 20.0
k = 1.0
v_r = -55.0
v_t = -40.0
eta = 0.0
a = 0.2
b = 0.025
d = 0.0
I_ext = 60.0
tau_s = 8.0

# connectivity parameters
W = random_connectivity(N, N, p, normalize=True)

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
thetas = f(N, loc=v_t, scale=Delta, lb=v_r, ub=2 * v_t - v_r)

# initialize the model
######################

# initialize operators
op = OperatorTemplate.from_yaml("config/ik_snn/ik_op")
exc_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1 / a, "b": b, "kappa": d,
            "g_e": 0.0, "E_i": E_i, "tau_s": tau_s, "v": v_r, "g_i": g, "E_e": E_e}

# initialize model
net = Network(dt, device=device)
net.add_diffeq_node("ik", f"config/ik_snn/ik", weights=W, source_var="s", target_var="s_i",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=exc_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    clear=True)

# simulation 1: steady-state
############################

# define input
inp = np.zeros((int(T/dt), N)) + I_ext
inp[int(start/dt):int(start/dt)+int(dur/dt), :] += amp

# perform steady-state simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False,
              cutoff=int(cutoff/dt))
s = obs.to_dataframe("out")
s.iloc[:, :] *= 1e3/tau_s

# figure settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 15.0
cmap = "ch:"

# create figure
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(nrows=2, ncols=4)

# plotting firing rate dynamics
ax = fig.add_subplot(grid[0, :3])
ax.plot(np.mean(s, axis=1), label=r"$\mathrm{mean}(r)$")
ax.plot(np.std(s, axis=1), label=r"$\mathrm{std}(r)$")
ax.axvline(x=start, color="black", linestyle="dashed")
ax.axvline(x=start+dur, color="black", linestyle="dashed", label="stimulus")
ax.set_xlim([cutoff, T])
ax.legend()
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$r$ (Hz)")
ax.set_title(f"Mean-Field Dynamics")

# plotting spikes
neurons = np.random.choice(N, size=200, replace=False)
ax2 = fig.add_subplot(grid[1, :3])
ax2.imshow(s.iloc[:, neurons].T, aspect="auto", interpolation="none", cmap="Greys")
xticks = [(tick-cutoff)*10 for tick in ax.get_xticks()]
xlabels = [int(tick) for tick in ax.get_xticks()]
ax2.set_xticks(xticks, labels=xlabels)
ax2.set_xlim([0, int((T-cutoff)/dts)])
ax2.set_xlabel("time (ms)")
ax2.set_ylabel("neurons")
ax2.set_title("Spikes")

# plotting connectivity
ax4 = fig.add_subplot(grid[0, 3])
im = ax4.imshow(W[neurons, :][:, neurons], aspect="auto", interpolation="none", cmap="viridis")
ax4.set_xlabel("neurons")
ax4.set_ylabel("neurons")
ax4.set_title(rf"$W$ for $p = {p}$")

# plotting connectivity
ax5 = fig.add_subplot(grid[1, 3])
ax5.hist(thetas[neurons], bins=10)
ax5.set_xlabel(r"spike threshold (mV)")
ax5.set_ylabel(r"count")
ax5.set_title(r"$\mathcal{N}(\bar \theta = $" + rf"${v_t}$" + r"$\mathrm{mV}, \sigma^2 = $" +
              f"${Delta}$" + r"$\mathrm{mV})$")

# save figure
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig("/home/richard-gast/Documents/data/dimensionality/figures/inh_dynamics1.pdf")

plt.show()
