from rectipy import Network, random_connectivity
from pyrates import OperatorTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from custom_functions import *

# define parameters
###################

# meta parameters
device = "cpu"
theta_dist = "gaussian"

# general model parameters
N = 1000
p = 0.1
E_e = 0.0
E_i = -60.0
v_spike = 40.0
v_reset = -55.0
g_in = 10.0

# get sweep condition
g = 0.75 #float(sys.argv[-1])
Delta = 1.0 #float(sys.argv[-2])

# input parameters
dt = 1e-2
dts = 1e-1
p_in = 0.3
dur = 20.0
amp = 30.0*1e-3
cutoff = 1000.0
T = 1000.0 + cutoff
start = 500.0 + cutoff

# model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
eta = 0.0
a = 0.03
b = -2.0
d = 50.0
s_e = 15.0*1e-3
tau_s = 6.0

# connectivity parameters
g_norm = g / np.sqrt(N * p)
W = random_connectivity(N, N, p, normalize=False)

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
thetas = f(N, loc=v_t, scale=Delta, lb=v_r, ub=2 * v_t - v_r)

# initialize the model
######################

# initialize operators
op = OperatorTemplate.from_yaml("config/ik_snn/ik_op")
exc_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1 / a, "b": b, "kappa": d,
            "g_e": g_norm, "E_i": E_i, "tau_s": tau_s, "v": v_t, "g_i": 0.0, "E_e": E_e}

# initialize model
net = Network(dt, device=device)
net.add_diffeq_node("ik", f"config/ik_snn/ik", weights=W, source_var="s", target_var="s_e",
                    input_var="g_e_in", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=exc_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    record_vars=["v"], clear=True)

# simulation 1: steady-state
############################

# define input
inp_neurons = np.random.choice(N, size=(int(N*p_in),), replace=False)
inp = np.zeros((int(T/dt), N))
inp[:, :] += poisson.rvs(mu=s_e*g_in*dt, size=(int(T/dt), N))
inp[int(start/dt):int(start/dt)+int(dur/dt), inp_neurons] += poisson.rvs(mu=amp * g_in * dt,
                                                                         size=(int(dur/dt), len(inp_neurons)))
inp = convolve_exp(inp, tau_s, dt)

# perform steady-state simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False,
              record_vars=[("ik", "v", True)], cutoff=int(cutoff/dt))
s = obs.to_dataframe("out")
s.iloc[:, :] *= 1e3/tau_s
v = obs.to_numpy(("ik", "v"))

# calculate neural correlation structure
window = 200.0
r_ss = s.loc[start-window:start, :]
r_ir = s.loc[start:start+window, :]
C_ss = get_cov(r_ss.values, center=True)
C_ir = get_cov(r_ir.values, center=True)

# calculate Kuramoto order parameter dynamics
r = np.mean(s.values, axis=1)
z = 1.0 - np.real(np.abs((1 - np.pi*C*r/k + 1.0j*(v-v_r))/(1 + np.pi*C*r/k - 1.0j*(v-v_r))))

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
fig = plt.figure(figsize=(12, 5))
grid = fig.add_gridspec(nrows=6, ncols=4)

# plotting firing rate dynamics
ax = fig.add_subplot(grid[:2, :3])
ax.plot(np.mean(s, axis=1), label=r"$\mathrm{mean}(r)$")
ax.plot(np.std(s, axis=1), label=r"$\mathrm{std}(r)$")
ax.axvline(x=start, color="black", linestyle="dashed")
ax.axvline(x=start+dur, color="black", linestyle="dashed", label="stimulus")
ax.set_xlim([cutoff, T])
ax.legend()
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$r$ (Hz)")
ax.set_title(f"Mean-field dynamics")

# plotting spikes
ax2 = fig.add_subplot(grid[2:4, :3])
spikes = s.values[:, inp_neurons].T
ax2.imshow(spikes / np.max(spikes, axis=1, keepdims=True), aspect="auto", interpolation="none", cmap="Greys")
xticks = [(tick-cutoff)*10 for tick in ax.get_xticks()]
xlabels = [int(tick) for tick in ax.get_xticks()]
ax2.set_xticks(xticks, labels=xlabels)
ax2.set_xlim([0, int((T-cutoff)/dts)])
ax2.set_xlabel("time (ms)")
ax2.set_ylabel("neurons")
ax2.set_title("Spiking dynamics in the network")

# plotting input
ax3 = fig.add_subplot(grid[4:, :3])
ax3.imshow(inp[int(cutoff/dt)::int(dts/dt), inp_neurons].T, aspect="auto", interpolation="none", cmap="viridis")
ax3.set_xticks(xticks, labels=xlabels)
ax3.set_xlim([0, int((T-cutoff)/dts)])
ax3.set_xlabel("time (ms)")
ax3.set_ylabel("neurons")
ax3.set_title("Input")

# plotting connectivity
ax4 = fig.add_subplot(grid[:3, 3])
im = ax4.imshow(W[inp_neurons, :][:, inp_neurons], aspect="auto", interpolation="none", cmap="cividis")
ax4.set_xlabel("neurons")
ax4.set_ylabel("neurons")
ax4.set_title(rf"$W$ for $p = {p}$")

# plotting spike thresholds
ax5 = fig.add_subplot(grid[3:, 3])
ax5.hist(thetas[inp_neurons], bins=10)
ax5.set_xlabel(r"spike threshold (mV)")
ax5.set_ylabel(r"count")
ax5.set_title(r"Spike threshold distribution")

# save figure
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig("/home/richard-gast/Documents/data/dimensionality/figures/exc_dynamics.pdf")

plt.show()
