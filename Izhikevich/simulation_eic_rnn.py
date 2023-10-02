from rectipy import Network, random_connectivity
from pyrates import CircuitTemplate, NodeTemplate
import sys
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import pickle


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

# general parameters
N = 100
p = 1.0
v_spike = 1e3
v_reset = -1e3

# RS neuron parameters
Ce = 100.0   # unit: pF
ke = 0.7  # unit: None
ve_r = -60.0  # unit: mV
ve_t = -40.0  # unit: mV
Delta_e = 0.5  # unit: mV
de = 100.0
ae = 0.03
be = -2.0
Ie = 60.0

# FS neuron parameters
Ci = 20.0   # unit: pF
ki = 1.0  # unit: None
vi_r = -55.0  # unit: mV
vi_t = -40.0  # unit: mV
Delta_i = 2.0  # unit: mV
di = 0.0
ai = 0.2
bi = 0.025
Ii = 45.0

# synaptic parameters
g_ampa = 1.0
g_gaba = 1.0
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 6.0
tau_gaba = 8.0
k_ee = 16.0
k_ei = 16.0
k_ie = 4.0
k_ii = 4.0

# define lorentzian of etas
spike_thresholds_e = lorentzian(N, eta=ve_t, delta=Delta_e, lb=ve_r, ub=2*ve_t - ve_r)
spike_thresholds_i = lorentzian(N, eta=vi_t, delta=Delta_i, lb=vi_r, ub=2*vi_t - vi_r)

# define connectivity
W_ee = random_connectivity(N, N, p, normalize=True)
W_ie = random_connectivity(N, N, p, normalize=True)
W_ei = random_connectivity(N, N, p, normalize=True)
W_ii = random_connectivity(N, N, p, normalize=True)

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), N))
inp[:int(0.5*cutoff/dt), :] += 20.0
inp[int(750/dt):int(2000/dt), :] -= 20.0

# run the model
###############

# initialize EIC node
eic_vars = {"C_e": Ce, "ke": ke, "v_r_e": ve_r, "v_theta_e": spike_thresholds_e, "eta_e": Ie, "tau_u_e": 1/ae, "be": be,
            "kappa": de, "g_e": g_ampa, "E_e": E_ampa, "g_i": g_gaba, "E_i": E_gaba, "tau_s_e": tau_ampa, "ve": ve_t,
            "C_i": Ci, "ki": ki, "v_r_i": vi_r, "v_theta_i": spike_thresholds_i, "eta_i": Ii, "tau_u_i": 1/ai, "bi": bi,
            "tau_s_i": tau_gaba, "vi": vi_t}

# construct EI circuit
neurons = {}
for i in range(N):
    eic = NodeTemplate.from_yaml("config/ik_snn/eic")
    for key, val in eic_vars.items():
        eic.update_var("eic_op", key, val if type(val) is float else val[i])
    neurons[f'eic_{i}'] = eic
neuron_keys = list(neurons.keys())
net = CircuitTemplate("eic", nodes=neurons)
net.add_edges_from_matrix(source_var="eic_op/se", target_var="eic_op/s_ee", weight=W_ee*k_ee, source_nodes=neuron_keys)
net.add_edges_from_matrix(source_var="eic_op/se", target_var="eic_op/s_ie", weight=W_ie*k_ie, source_nodes=neuron_keys)
net.add_edges_from_matrix(source_var="eic_op/si", target_var="eic_op/s_ei", weight=W_ei*k_ei, source_nodes=neuron_keys)
net.add_edges_from_matrix(source_var="eic_op/si", target_var="eic_op/s_ii", weight=W_ii*k_ii, source_nodes=neuron_keys)

# initialize rectipy model
model = Network(dt=dt, device="cuda:0")
model.add_diffeq_node("eic_net", node=net, input_var="I_ext_i", output_var="se",
                      spike_var=["spike_e", "spike_i"], spike_def=["ve", "vi"], spike_reset=v_reset,
                      spike_threshold=v_spike, verbose=True, clear=False, op="eic_op", N=N)

# perform simulation
obs = model.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=True)
res = obs.to_numpy("out")
rs_res = np.mean(res[:, :N], axis=1)
fs_res = np.mean(res[:, N:], axis=1)

# plot results
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot()
ax.plot(rs_res, color="royalblue", label="rs")
ax.plot(fs_res, color="darkorange", label="fs")
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel('time (ms)')
ax.set_title('mean-field dynamics')
ax.legend()
plt.show()

# save results
pickle.dump({'rs_results': rs_res, 'fs_results': fs_res},
            open("results/eic_snn_het_fs_oscillatory.p", "wb"))
