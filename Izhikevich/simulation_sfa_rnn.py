import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu import RNN
import matplotlib.pyplot as plt
import pickle
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


def ik_ata(y: np.ndarray, N: int, inp: np.ndarray, v_r: float, v_t: np.ndarray, k: float, E_r: float, C: float,
           J: float, g: float, tau_s: float, b: float, a: float, d: float, v_spike: float, v_reset: float,
           dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities."""

    # extract state variables from u
    v, u, s = y[:N], y[N:2*N], y[2*N]

    # calculate network input
    spikes = v >= v_spike
    rates = np.mean(spikes / dt)

    # calculate vector field of the system
    dv = (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + g*s*(E_r - v) - u)/C
    du = a*(b*(v-v_r) - u)
    ds = J*rates - s/tau_s

    # update state variables
    v_new = v + dt * dv
    u_new = u + dt * du
    s_new = s + dt * ds

    # reset membrane potential and apply spike frequency adaptation
    v_new[spikes] = v_reset
    u_new[spikes] += d

    # store updated state variables
    y[:N] = v_new
    y[N:2*N] = u_new
    y[2*N] = s_new
    y[2*N+1] = rates

    return y

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
spike_thresholds = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=0.0)

# define inputs
T = 5500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 40.0
inp[int(1000/dt):int(5000/dt)] += np.linspace(0.0, 30.0, num=int(4000/dt))
# inp[int(3000/dt):int(5000/dt)] += np.linspace(30.0, 0.0, num=int(2000/dt))

# run the model
###############

# initialize model
u_init = np.zeros((2*N+2,))
u_init[:N] -= 60.0
model = RNN(N, 2*N+2, ik_ata, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset, d=d, a=a, b=b,
            tau_s=tau_s, J=J, g=g, E_r=E_r, u_init=u_init)

# define outputs
outputs = {'r': {'idx': np.asarray([2*N+1]), 'avg': False}, 'u': {'idx': np.arange(N, 2*N), 'avg': True}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, parallel=True, fastmath=True)

# plot results
fig, ax = plt.subplots(nrows=2, figsize=(12, 6))
ax[0].plot(np.mean(res["r"], axis=1))
ax[0].set_ylabel(r'$r(t)$')
ax[1].plot(np.mean(res["u"], axis=1))
ax[1].set_ylabel(r'$u(t)$')
ax[1].set_xlabel('time')
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/sfa_rnn_high.p", "wb"))