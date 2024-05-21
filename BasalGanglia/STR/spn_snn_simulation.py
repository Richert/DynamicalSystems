import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(2)
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
device = "cuda:0"

# model parameters
C = 15.0
k = 1.0
v_r = -80.0
v_t = -30.0
v_spike = 15.0
v_reset = -90.0
Delta = 1.0
kappa = 90.0
a = 0.01
b = -20.0
tau_s = 4.0
g_i = 5.0
E_i = -60.0
W = np.load('config/msn_conn.npy')

# define lorentzian of spike thresholds
spike_thresholds = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2*v_t-v_r)

# define inputs
T = 100.0
cutoff = 0.0
start = 30.0
stop = 60.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt), 1)) + 900.0
# inp[int(start/dt):int(stop/dt)] += 50.0

# collect parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": spike_thresholds, "tau_u": 1/a, "b": b, "kappa": kappa,
             "g_i": g_i, "E_i": E_i, "tau_s": tau_s, "v": v_t}

# run the model
###############

# initialize model
net = Network(dt=dt, device=device)
net.add_diffeq_node("SPNs", node=f"config/snn/ik", weights=W, source_var="s", target_var="s_i",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=node_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True)

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False)
res = obs.to_numpy("out")

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
im = ax.imshow(res.T, aspect=0.25, interpolation="none")
plt.colorbar(im, ax=ax, shrink=0.65)
ax.set_xlabel('time')
ax.set_ylabel('neurons')
ax.set_title("Spiking activity")
ax = axes[1]
ax.plot(np.mean(res, axis=1))
ax.set_xlabel("time")
ax.set_ylabel("s")
ax.set_title("Average synaptic activation")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'results': res}, open("results/spn_rnn.p", "wb"))
