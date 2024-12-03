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
g = 1.5 #float(sys.argv[-1])
Delta = 1.0 #float(sys.argv[-2])

# input parameters
dt = 1e-2
dts = 1e-1
cutoff = 1000.0
T = 1000.0 + cutoff

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
inp = np.zeros((int(T/dt), N))
inp[:, :] += poisson.rvs(mu=s_e*g_in*dt, size=(int(T/dt), N))
inp = convolve_exp(inp, tau_s, dt)

# perform steady-state simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False,
              record_vars=[("ik", "v", True)], cutoff=int(cutoff/dt))
s = obs.to_dataframe("out")
s.iloc[:, :] *= 1e3/tau_s
v = obs.to_numpy(("ik", "v"))

# calculate correlation between inputs and neurons
inp = inp[int(cutoff/dt)::int(dts/dt), :]
spikes = s.values
C_net = get_cov(spikes, center=True)
C_inp = get_cov(inp, center=True)
eigvals_net = np.real(np.linalg.eigvals(C_net))
dim_net = np.sum(eigvals_net)**2/np.sum(eigvals_net**2)
eigvals_inp = np.real(np.linalg.eigvals(C_inp))
dim_inp = np.sum(eigvals_inp)**2/np.sum(eigvals_inp**2)

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
fig = plt.figure(figsize=(6, 4))
grid = fig.add_gridspec(nrows=2, ncols=3)

# plotting spiking dynamics
neurons = np.random.choice(N, size=(100,))
ax1 = fig.add_subplot(grid[0, :])
spikes = spikes[:, neurons].T
ax1.imshow(spikes / np.max(spikes, axis=1, keepdims=True), aspect="auto", interpolation="none", cmap="Greys")
ax1.set_xlabel("time steps")
ax1.set_ylabel("neurons")

# plotting firing rate distributions
window = int(200.0/dts)
ax2 = fig.add_subplot(grid[1, 0])
_, bins, _ = ax2.hist(np.mean(s.values, axis=0), bins=10, label="SS", alpha=0.5)
ax2.set_xlabel(r"$r$ (Hz)")
ax2.set_ylabel(r"count")
ax2.set_title("firing rates")

# plotting covariances
ax3 = fig.add_subplot(grid[1, 1])
C_net[np.eye(N) == 1.0] = 0.0
im = ax3.imshow(C_net[neurons, :][:, neurons], aspect="auto", cmap="viridis", interpolation="none")
ax3.set_xlabel(r"neurons")
ax3.set_ylabel(r"neurons")
ax3.set_title(rf"Network: $D(C) = {np.round(dim_net, decimals=0)}$")

# plotting covariances
ax4 = fig.add_subplot(grid[1, 2])
C_inp[np.eye(N) == 1.0] = 0.0
im = ax4.imshow(C_inp[neurons, :][:, neurons], aspect="auto", cmap="viridis", interpolation="none")
ax4.set_xlabel(r"neurons")
ax4.set_ylabel(r"neurons")
ax4.set_title(rf"Input: $D(C) = {np.round(dim_inp, decimals=0)}$")

# save figure
fig.suptitle(rf"$\Delta = {np.round(Delta, decimals=1)}$ mV")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig("/home/richard-gast/Documents/data/dimensionality/figures/exc_high_hom.pdf")

plt.show()
