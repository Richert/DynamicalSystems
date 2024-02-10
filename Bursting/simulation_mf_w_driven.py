from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb
from scipy.interpolate import interp1d

# define parameters
###################

# condition
cond = "no_sfa_1"
model = "ik_eta_corrected"
op = "eta_op_corrected"
cond_map = {
    "no_sfa_1": {"kappa": 0.0, "eta": 0.0, "eta_inc": 30.0, "eta_init": -30.0, "b": -5.0, "delta": 5.0},
    "weak_sfa_1": {"kappa": 100.0, "eta": 0.0, "eta_inc": 35.0, "eta_init": 0.0, "b": -5.0, "delta": 5.0},
    "strong_sfa_1": {"kappa": 300.0, "eta": 0.0, "eta_inc": 45.0, "eta_init": 0.0, "b": -5.0, "delta": 5.0},
    "no_sfa_2": {"kappa": 0.0, "eta": -150.0, "eta_inc": 190.0, "eta_init": -50.0, "b": -20.0, "delta": 5.0},
    "weak_sfa_2": {"kappa": 100.0, "eta": -20.0, "eta_inc": 70.0, "eta_init": -100.0, "b": -20.0, "delta": 5.0},
    "strong_sfa_2": {"kappa": 300.0, "eta": 40.0, "eta_inc": 100.0, "eta_init": 0.0, "b": -20.0, "delta": 5.0},
}

# model parameters
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
eta = 0.0  # unit: pA
Delta = cond_map[cond]["delta"]
kappa = cond_map[cond]["kappa"]
tau_u = 35.0
b = cond_map[cond]["b"]
tau_s = 6.0
tau_x = 300.0
g = 15.0
E_r = 0.0

# define inputs
T = 7000.0
dt = 1e-2
dts = 1e-1
cutoff = 1000.0
inp = np.zeros((int(T/dt),)) + cond_map[cond]["eta"]
inp[:int(300.0/dt)] += cond_map[cond]["eta_init"]
inp[int(2000/dt):int(5000/dt),] += cond_map[cond]["eta_inc"]

# load ws
w = pickle.load(open(f"results/snn_etas_{cond}.pkl", "rb"))["results"]["u_width"]
time = np.linspace(0, T, num=int(T/dts))
w_interp = interp1d(time, w, kind="cubic")
w_in = np.zeros((int(T/dt),))
for i, t in enumerate(np.linspace(0, T, num=int(T/dt))):
    w_in[i] = w_interp(t)

# run the model
###############

# initialize model
ik = CircuitTemplate.from_yaml(f"config/mf/{model}")

# update parameters
node_vars = {'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'kappa': kappa, 'tau_u': tau_u, 'b': b,
             'tau_s': tau_s, 'g': g, 'E_r': E_r, 'tau_x': tau_x, 'eta': eta}
ik.update_var(node_vars={f"p/{op}/{key}": val for key, val in node_vars.items()})

# run simulation
res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
             outputs={'s': f'p/{op}/s', 'u': f'p/{op}/u'}, inputs={f'p/{op}/I_ext': inp, f'p/{op}/w': w},
             decorator=nb.njit, fastmath=True, float_precision="float64")

# save results to file
pickle.dump({"results": res, "params": node_vars}, open(f"results/mf_etas_global_{cond}.pkl", "wb"))

# plot results
fig, ax = plt.subplots(nrows=2, figsize=(12, 5))
ax[0].plot(res["s"])
ax[0].set_ylabel(r'$s(t)$')
ax[1].plot(res["u"])
ax[1].set_ylabel(r'$u(t)$')
ax[1].set_xlabel("time (ms)")
plt.tight_layout()
plt.show()
