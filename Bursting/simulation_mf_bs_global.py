from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb

# define parameters
###################

# condition
cond = "low_kappa"

# model parameters
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
eta = 0.0  # unit: pA
Delta = 2.5
kappa = 0.0 if cond == "low_kappa" else 1.0
tau_u = 35.0
b = -8.0
tau_s = 6.0
tau_x = 300.0
g = 15.0
E_r = 0.0

# define inputs
T = 6000.0
dt = 1e-2
dts = 1e-1
cutoff = 1000.0
inp = np.zeros((int(T/dt),)) + (-15.0 if cond == "low_kappa" else 15.0)
# inp[:int(200.0/dt)] -= 10.0
inp[int(2000/dt):int(4000/dt),] += (25.0 if cond == "low_kappa" else 15.0)

# run the model
###############

# initialize model
ik = CircuitTemplate.from_yaml("config/mf/recovery_b_global")

# update parameters
node_vars = {'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'kappa': kappa, 'tau_u': tau_u, 'b': b,
             'tau_s': tau_s, 'g': g, 'E_r': E_r, 'tau_x': tau_x, 'eta': eta}
ik.update_var(node_vars={f"p/global_recovery_b_op/{key}": val for key, val in node_vars.items()})

# run simulation
res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
             outputs={'s': 'p/global_recovery_b_op/s', 'u': 'p/global_recovery_b_op/u'},
             inputs={'p/global_recovery_b_op/I_ext': inp},
             decorator=nb.njit, fastmath=True, float_precision="float64")

# save results to file
file_num = "" if cond == "low_kappa" else "2"
pickle.dump({"results": res, "params": node_vars}, open(f"results/mf_bs_global{file_num}.pkl", "wb"))

# plot results
fig, ax = plt.subplots(nrows=2, figsize=(12, 5))
ax[0].plot(res["s"])
ax[0].set_ylabel(r'$s(t)$')
ax[1].plot(res["u"])
ax[1].set_ylabel(r'$u(t)$')
ax[1].set_xlabel("time (ms)")
plt.tight_layout()
plt.show()
