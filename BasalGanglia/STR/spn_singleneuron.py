import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(2)
import numpy as np
from rectipy import Network
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'TkAgg'


# define parameters
###################

N = 1
device = "cpu"

# model parameters
C = 50.0
k = 1.0
v_r = -80.0
v_t = -30.0
v_spike = 40.0
v_reset = -55.0
Delta = 0.8
eta = 0.0
kappa = 250.0
a = 0.01
b = -20.0
tau_s = 8.0
g_i = 4.0
E_i = -60.0

# define inputs
cutoff = 100.0
T = 2100.0 + cutoff
starts = [300.0, 900.0, 1500.0]
amps = [350.0, 500.0, 650.0]
dur = 300.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt), 1))
for start, amp in zip(starts, amps):
    inp[int((start+cutoff)/dt):int((start+dur+cutoff)/dt), 0] += amp

# collect parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": v_t, "tau_u": 1/a, "b": b, "kappa": kappa,
             "g_i": g_i, "E_i": E_i, "tau_s": tau_s, "v": v_t, "eta": eta}

# run the model
###############

# initialize model
net = Network(dt=dt, device=device)
net.add_diffeq_node("SPNs", node=f"config/snn/ik", source_var="s", target_var="s_i",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=node_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True, record_vars=["v"], N=N)

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, cutoff=int(cutoff/dt),
              record_vars=[("SPNs", "v", False)]
              )
s = obs.to_numpy("out")
v = obs.to_numpy(("SPNs", "v"))

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
ax.plot(v)
ax.set_xlabel('time')
ax.set_ylabel('v')
ax = axes[1]
ax.plot(s)
ax.set_xlabel("time")
ax.set_ylabel("s")
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'results': res}, open("results/spn_rnn.p", "wb"))
