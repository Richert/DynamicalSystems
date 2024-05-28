import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(2)
import pickle
import numpy as np
from rectipy import Network
import matplotlib.pyplot as plt
from scipy.stats import cauchy
plt.rcParams['backend'] = 'TkAgg'


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


# define parameters
###################

N = 1000
device = "cpu"

# model parameters
C = 50.0
k = 1.0
v_r = -80.0
v_t = -30.0
eta = 400.0
Delta = 0.35
kappa = 150.0
a = 0.01
b = -20.0
tau_s = 4.0
g_i = 4.0
E_i = -60.0
v_spike = 500.0
v_reset = -500.0
# W = np.load('config/msn_conn.npy')
v_t1 = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2*v_t-v_r)
v_t2 = lorentzian(N, eta=v_t, delta=Delta - 0.25, lb=v_r, ub=2*v_t-v_r)

# define inputs
cutoff = 600.0
T = 900.0 + cutoff
start = 300.0 + cutoff
stop = 600.0 + cutoff
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), N))
for step in range(inp.shape[0]):
    v_t = v_t2 if int(start/dt) <= step < int(stop/dt) else v_t1
    inp[step, :] = v_t

# collect parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "tau_u": 1/a, "b": b, "kappa": kappa,
             "g_i": g_i, "E_i": E_i, "tau_s": tau_s, "v": v_t, "eta": eta}

# run the model
###############

# initialize model
net = Network(dt=dt, device=device)
net.add_diffeq_node("SPNs", node=f"config/snn/ik", source_var="s", target_var="s_i",
                    input_var="v_theta", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=node_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True, N=N, record_vars=["v"])

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, cutoff=int(cutoff/dt),
              record_vars=[("SPNs", "v", True)])
s = obs.to_numpy("out")
v = obs.to_numpy(("SPNs", "v"))

# save results
pickle.dump({'s': s, 'v': v}, open("results/spn_snn.pkl", "wb"))

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
im = ax.imshow(s.T, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax, shrink=0.65)
ax.set_xlabel('time')
ax.set_ylabel('neurons')
ax.set_title("Spiking activity")
ax = axes[1]
ax.plot(np.mean(s, axis=1) / tau_s)
ax.set_xlabel("time")
ax.set_ylabel("r")
ax.set_title("Average synaptic activation")
plt.tight_layout()
plt.show()
