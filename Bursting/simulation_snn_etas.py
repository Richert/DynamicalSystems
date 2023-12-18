import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import pickle
import numpy as np
from rectipy import Network, random_connectivity
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# condition
cond = "strong_sfa"
cond_map = {
    "no_sfa": {"kappa": 0.0, "eta": 20.0, "eta_inc": 10.0},
    "weak_sfa": {"kappa": 0.2, "eta": 37.0, "eta_inc": 10.0},
    "strong_sfa": {"kappa": 0.4, "eta": 37.0, "eta_inc": 10.0}
}

# model parameters
N = 2000
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
eta = 0.0  # unit: pA
Delta = 10.0
kappa = cond_map[cond]["kappa"]
tau_u = 35.0
b = -2.0
tau_s = 6.0
tau_x = 300.0
g = 15.0
E_r = 0.0

v_reset = -2000.0
v_peak = 2000.0

# define inputs
T = 7000.0
dt = 1e-2
dts = 1e-1
cutoff = 1000.0
inp = np.zeros((int(T/dt), 1)) + cond_map[cond]["eta"]
# inp[:int(200.0/dt)] -= 10.0
inp[int(2000/dt):int(5000/dt), 0] += cond_map[cond]["eta_inc"]

# define lorentzian distribution of etas
etas = eta + Delta * np.tan(0.5*np.pi*(2*np.arange(1, N+1)-N-1)/(N+1))

# define connectivity
# W = random_connectivity(N, N, 0.2)

# run the model
###############

# initialize model
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": v_t, "eta": etas, "tau_u": tau_u, "b": b, "kappa": kappa,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_r, "tau_x": tau_x}

# initialize model
net = Network(dt=dt, device="cpu")
net.add_diffeq_node("sfa", f"config/snn/recovery", #weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=node_vars.copy(), op="recovery_op", spike_reset=v_reset, spike_threshold=v_peak,
                    verbose=False, clear=True, N=N, float_precision="float64")

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), verbose=True, cutoff=int(cutoff/dt))
res = obs.to_dataframe("out")

# save results to file
pickle.dump({"results": res, "params": node_vars}, open(f"results/snn_etas_{cond}.pkl", "wb"))

# plot results
fig, ax = plt.subplots(nrows=2, figsize=(12, 6))
spikes = res.values
ax[0].imshow(spikes.T, interpolation="none", cmap="Greys", aspect="auto")
ax[0].set_ylabel(r'neuron id')
ax[1].plot(res.index, np.mean(spikes, axis=1))
ax[1].set_ylabel(r'$s(t)$')
ax[1].set_xlabel('time')
plt.tight_layout()
plt.show()
