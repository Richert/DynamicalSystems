from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb
from rectipy import Network, random_connectivity
import sys
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
from reservoir_computing.utility_funcs import lorentzian
from scipy.ndimage import gaussian_filter1d

# define parameters
###################

# model parameters
N = 200
p = 0.1
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 0.8
eta = 50.0
a = 0.03
b = -2.0
d = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1
noise_lvl = 20.0
noise_sigma = 100.0
inp = np.zeros((int(T/dt), N)) + eta
for i in range(N):
    noise = noise_lvl * np.random.randn(inp.shape[0])
    noise = gaussian_filter1d(noise, sigma=noise_sigma)
    inp[:, i] += noise

# run the mean-field model
##########################

# initialize model
ik = CircuitTemplate.from_yaml("config/ik_mf/ik")

# update parameters
ik.update_var(node_vars={'p/ik_op/C': C, 'p/ik_op/k': k, 'p/ik_op/v_r': v_r, 'p/ik_op/v_t': v_t, 'p/ik_op/Delta': Delta,
                         'p/ik_op/d': d, 'p/ik_op/a': a, 'p/ik_op/b': b, 'p/ik_op/tau_s': tau_s, 'p/ik_op/g': g,
                         'p/ik_op/E_r': E_r})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
                outputs={'s': 'p/ik_op/s'}, inputs={'p/ik_op/I_ext': inp[:, 0]},
                decorator=nb.njit)

# run the SNN model
###################

# define lorentzian of etas
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=0.0)

# define connectivity
W = random_connectivity(N, N, p, normalize=True)

# initialize model
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "tau_u": 1/a, "b": b, "kappa": d,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

# initialize model
net = Network(dt=dt, device="cpu")
net.add_diffeq_node("snn", f"config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True, device="cuda:0")

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=True, cutoff=int(cutoff/dt))
res_snn = obs.to_dataframe("out")

# plot results
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(res_mf.index, res_mf["s"], label="FRE")
ax.plot(res_mf.index, np.mean(res_snn, axis=1), label="SNN")
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel("time (ms)")
plt.legend()
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
