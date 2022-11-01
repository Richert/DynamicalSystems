import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu import RNN
from pyrecu.neural_models import ik_spike_reset2
import matplotlib.pyplot as plt
import pickle
from scipy.stats import cauchy
from typing import Union, Callable
plt.rcParams['backend'] = 'TkAgg'


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def ik_ata(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple, v_r: float,
           v_t: np.ndarray, k: float, E_r: float, C: float, g: float, tau_s: float, b: float, a: float,
           q: float, J: float) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities."""

    dy = np.zeros_like(y)

    # extract state variables from u
    m = 2*N
    v, u, s = y[:N], y[N:m], y[m]

    # retrieve extrinsic input at time t
    inp = infunc(t, *inargs)

    # calculate vector field of the system
    dy[:N] = (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + g*s*(E_r - v) + q*(np.mean(v)-v) - u)/C
    dy[N:m] = a*(b*(v-v_r) - u)
    dy[m] = -s/tau_s + J*np.mean(rates)

    return dy


# define parameters
###################

# model parameters
N = 10000
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
v_spike = 2000.0  # unit: mV
v_reset = -3000.0  # unit: mV
Delta = 1.0  # unit: mV
d = 100.0
a = 0.03
b = -2.0
tau_s = 6.0
J = 1.0
g = 15.0
E_r = 0.0

# define lorentzian of etas
spike_thresholds = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=v_r-v_t)

# define inputs
T = 5500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 40.0
inp[int(1000/dt):int(5000/dt)] += np.linspace(0.0, 30.0, num=int(4000/dt))
#inp[int(3000/dt):int(5000/dt)] += np.linspace(30.0, 0.0, num=int(2000/dt))

# run the model
###############

# initialize model
u_init = np.zeros((2*N+1,))
u_init[:N] -= 60.0
run_args = (v_r, spike_thresholds, k, E_r, C, g, tau_s, b, a, 0.0, J)
spike_args = (v_spike, v_reset, d)
model = RNN(N, 2*N+2, ik_ata, evolution_args=run_args, callback_func=ik_spike_reset2, callback_args=spike_args,
            u_init=u_init)

# define outputs
outputs = {'s': {'idx': np.asarray([2*N]), 'avg': False}, 'u': {'idx': np.arange(N, 2*N), 'avg': True}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, parallel=True, fastmath=True)

# plot results
fig, ax = plt.subplots(nrows=2, figsize=(12, 6))
ax[0].plot(np.mean(res["s"], axis=1))
ax[0].set_ylabel(r'$s(t)$')
ax[1].plot(np.mean(res["u"], axis=1))
ax[1].set_ylabel(r'$u(t)$')
ax[1].set_xlabel('time')
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/sfa_rnn_high.p", "wb"))
