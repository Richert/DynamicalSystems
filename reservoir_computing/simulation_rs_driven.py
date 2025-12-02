from rectipy import Network, random_connectivity
import numpy as np
from utility_funcs import lorentzian
import matplotlib.pyplot as plt

# network definition
####################

# network parameters
N = 1000
p = 0.1
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.0
eta = 55.0
a = 0.03
b = -2.0
d = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# simulation parameters
cutoff = 10000.0
T = 20000.0 + cutoff
dt = 1e-2
sr = 100
steps = int(np.round(T/dt))
time = np.linspace(0.0, T, num=steps)

# input definition
alpha = 0.0007
omega = 5.0
p_in = 0.32

# initialize input
I_ext = np.zeros((steps, 1))
I_ext[:, 0] = np.sin(2.0 * np.pi * omega * time * 1e-3)
ko = I_ext[::sr, 0]

# create connectivity matrix
W = random_connectivity(N, N, p, normalize=True)

# create input matrix
W_in = np.zeros((N, 1))
idx = np.random.choice(np.arange(N), size=int(N*p_in), replace=False)
W_in[idx, 0] = alpha

# create background current distribution
thetas = lorentzian(N, v_t, Delta, v_r, 2 * v_t - v_r)

# collect remaining model parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

# initialize model
net = Network(dt=dt, device="cpu")
net.add_diffeq_node("rs", f"config/ik/rs", weights=W, source_var="s", target_var="s_in",
                        input_var="s_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                        node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                        dt=dt, clear=True)
net.add_func_node("inp", 1, activation_function="identity")
net.add_edge("inp", "rs", weights=W_in, train=None)

# simulation
obs = net.run(inputs=I_ext, sampling_steps=sr, enable_grad=False, verbose=True)
res = obs.to_dataframe("out")
time = np.linspace(0, T, num=res.shape[0])
ik_inp = np.mean(res.loc[:, W_in[:, 0] > 0].values, axis=-1)
ik_noinp = np.mean(res.loc[:, W_in[:, 0] < alpha].values, axis=-1)

# plot results
start = 5000
stop = 10000
fig, axes = plt.subplots(nrows=4, figsize=(12, 8))
ax = axes[0]
ax.imshow(res.values.T ,aspect="auto", interpolation="none", cmap="Greys")
ax.set_xlabel("time (steps)")
ax.set_ylabel("neurons")
ax.set_title("SNN dynamics")
ax = axes[1]
ax.plot(time[start:stop], ik_inp[start:stop])
ax.set_xlabel("time (ms)")
ax.set_ylabel("$s$")
ax.set_title("Mean-field of driven cells")
ax = axes[2]
ax.plot(time[start:stop], ik_noinp[start:stop])
ax.set_xlabel("time (ms)")
ax.set_ylabel("$s$")
ax.set_title("Mean-field of non-driven cells")
ax = axes[3]
ax.plot(time[start:stop], ko[start:stop])
ax.set_xlabel("time (ms)")
ax.set_ylabel(r"$I_{ext}$")
ax.set_title("Driver")
plt.suptitle(fr"$\omega = {omega}$, $\alpha = {alpha}$, $p_i = {p_in}$")

plt.tight_layout()
# plt.savefig(f'results/snn_entrainment_9.pdf')
plt.show()
