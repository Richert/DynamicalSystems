from pyrecu import RNN
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
from numpy import pi, sqrt, arctan
from numba import njit


@njit(fastmath=True)
def correct_input(inp: float, k: float, v_r: float, v_t: float, v_spike: float, v_reset: float, g: float, s: float,
                  E: float, u: float, Delta: float):
    alpha = v_r + v_t + g*s/k
    mu = 4*(v_r*v_t + (inp - u*np.pi**(2/3) + g*s*E)/k) - alpha**2
    if mu > 0:
        mu_sqrt = sqrt(mu)
        inp_c = pi**2*k*mu/(4*(arctan((2*v_spike-alpha)/mu_sqrt) - arctan((2*v_reset-alpha)/mu_sqrt))**2)
        inp_c += k*alpha**2/4 + u*np.pi**(2/3)
        inp_c -= k*v_r*v_t + g*s*E
    else:
        inp_c = inp
    return inp_c


def ik_run(t, y, N, spikes, in_func, in_args, v_t, v_r, k, J, g, Delta, C, v_z, v_p, E_r, b, a, d, tau_s, dt):

    dy = np.zeros_like(y)

    r = y[0]
    v = y[1]
    u = y[2]
    s = y[3]
    r_in = J*r
    inp = in_func(t, *in_args)
    I_ext = correct_input(inp, k, v_r, v_t, v_p, v_z, g, s, E_r, u, Delta)

    dy[0] = (r * (-g * s + k * (2.0 * v - v_r - v_t)) + Delta * k ** 2 * (v - v_r) / (pi * C)) / C
    dy[1] = (-pi * C * r * (pi * C * r / k + Delta) + I_ext + g * s * (
                E_r - v) + k * v * (v - v_r - v_t) + k * v_r * v_t - u) / C
    dy[2] = a * (b * (v - v_r) - u) + d * r
    dy[3] = r_in - s / tau_s

    return dy


def cb(u, N, *args):
    return u, np.zeros((N,))


# define parameters
###################

# model parameters
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
v_spike = np.inf  # unit: mV
v_reset = -np.inf  # unit: mV
Delta = 0.5  # unit: mV
d = 0.0
a = 0.003
b = 0.0
tau_s = 6.0
g = 15.0
J = 1.0
E_r = 0.0

# define inputs
T = 4500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 50.0
inp[int(1500/dt):int(2500/dt)] = 70.0
inp[int(2500/dt):int(3500/dt)] = 27.5

model_params = (v_t, v_r, k, J, g, Delta, C, v_reset, v_spike, E_r, b, a, d, tau_s, dt)

# run the model
###############

# initialize model
u_init = np.zeros((4,))
u_init[1] = -60.0
model = RNN(1, 4, ik_run, evolution_args=model_params, callback_func=cb, callback_args=(), u_init=u_init)

# define outputs
outputs = {'s': {'idx': np.asarray([3]), 'avg': False}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, fastmath=True)

# plot results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(np.mean(res["s"], axis=1))
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel('time')
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/spike_mech_fre_inf.p", "wb"))
