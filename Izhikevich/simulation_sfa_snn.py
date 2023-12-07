import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from rectipy import Network, random_connectivity
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

# model parameters
N = 1000
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
eta = 0.0  # unit: pA
Delta = 0.5  # unit: pA
kappa = 0.0
tau_u = 35.0
b = -2.0
tau_s = 6.0
tau_x = 250.0
g = 15.0
E_r = 0.0
v_reset = -1000.0
v_peak = 1000.0

# define inputs
T = 3500.0
dt = 1e-2
dts = 1e-1
cutoff = int(500.0/dt)
inp = np.zeros((int(T/dt), 1)) + 45.0
inp[:int(cutoff/dt)] -= 30.0
inp[int(1000/dt):int(2000/dt),] += 100.0

# define lorentzian of etas
bs = lorentzian(N, eta=b, delta=Delta, lb=b-10*Delta, ub=b+10*Delta)

# define connectivity
W = random_connectivity(N, N, 0.2)

# run the model
###############

# initialize model
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": v_t, "eta": eta, "tau_u": tau_u, "b": bs, "kappa": kappa,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_r, "tau_x": tau_x}

# initialize model
net = Network(dt=dt, device="cpu")
net.add_diffeq_node("sfa", f"config/ik_snn/sfa", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=node_vars.copy(), op="sfa_op", spike_reset=v_reset, spike_threshold=v_peak,
                    verbose=False, clear=True)

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), verbose=True, cutoff=cutoff)
res = obs.to_dataframe("out")

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
