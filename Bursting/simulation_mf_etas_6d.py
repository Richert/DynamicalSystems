from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb

# define parameters
###################

# condition
cond = "high_sfa"
model = "ik_eta_6d"
op = "eta_op_6d"
cond_map = {
    "low_sfa": {"kappa": 30.0, "eta": 100.0, "eta_inc": 30.0, "eta_init": -30.0, "b": 5.0, "delta": 5.0},
    "med_sfa": {"kappa": 100.0, "eta": 120.0, "eta_inc": 30.0, "eta_init": 0.0, "b": 5.0, "delta": 5.0},
    "high_sfa": {"kappa": 300.0, "eta": 60.0, "eta_inc": -35.0, "eta_init": 0.0, "b": -7.5, "delta": 5.0},
    "low_delta": {"kappa": 0.0, "eta": -125.0, "eta_inc": 135.0, "eta_init": -30.0, "b": -15.0, "delta": 1.0},
    "med_delta": {"kappa": 0.0, "eta": 100.0, "eta_inc": 30.0, "eta_init": -30.0, "b": 5.0, "delta": 5.0},
    "high_delta": {"kappa": 0.0, "eta": 6.0, "eta_inc": -40.0, "eta_init": 30.0, "b": -6.0, "delta": 10.0},
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
             outputs={'s': f'p/{op}/s', 'u': f'p/{op}/u', 'v': f'p/{op}/v', 'r': f'p/{op}/r',
                      'x': f'p/{op}/x', 'w': f'p/{op}/w'}, clear=False,
             inputs={f'p/{op}/I_ext': inp}, float_precision="complex64", decorator=nb.njit)

# collect results
time = res.index
r_mf = res["r"].values
v_mf = res["v"].values
u_mf = res["u"].values
w_mf = res["w"].values
x_mf = res["x"].values
s_mf = res["s"].values

# calculate width of v and u
y = np.sqrt(k/C)*(v_mf-v_r)
x = np.pi*C*r_mf/k
z = np.abs((1 - x + 1.0j*y)/(1 + x - 1.0j*y))
u_delta = (b*np.pi*C*r_mf/k - kappa*x_mf)*z**2
v_delta = np.pi*C*r_mf/k

# plot distribution dynamics for MF
fig, ax = plt.subplots(nrows=4, figsize=(12, 7))
ax[0].plot(time, v_mf, color="royalblue")
ax[0].fill_between(time, v_mf - v_delta, v_mf + v_delta, alpha=0.3, color="royalblue", linewidth=0.0)
ax[0].set_title("v (mV)")
ax[1].plot(time, u_mf, color="darkorange")
ax[1].fill_between(time, u_mf - u_delta, u_mf + u_delta, alpha=0.3, color="darkorange", linewidth=0.0)
ax[1].set_title("u (pA)")
ax[2].plot(time, s_mf, color="black")
ax[2].set_title("s (dimensionless)")
ax[2].set_xlabel("time (ms)")
ax[3].plot(time, z, color="red")
ax[3].set_title("z")
ax[3].set_xlabel("time (ms)")
fig.suptitle("MF")
plt.tight_layout()
plt.show()
