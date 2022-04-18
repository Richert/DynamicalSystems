import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu import RNN
from pyrecu.neural_models import ik_ata
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


# define parameters
###################

N = 10000

# model parameters
C = 20.0   # unit: pF
k = 1.0  # unit: None
v_r = -55.0  # unit: mV
v_t = -40.0  # unit: mV
v_spike = 500.0  # unit: mV
v_reset = -900.0  # unit: mV
Delta = 0.5  # unit: mV
d = 0.0
a = 0.2
b = 0.025
tau_s = 6.0
J = 1.0
g = 15.0
q = 0.0
E_r = -65.0

# define lorentzian of etas
spike_thresholds = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=0.0)

# define inputs
T = 1500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 60.0
inp[int(800/dt):int(1200/dt)] += 60.0

# run the model
###############

# initialize model
u_init = np.zeros((3*N,))
u_init[:N] += v_r
model = RNN(N, 3*N, ik_ata, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset, d=d, a=a, b=b,
            tau_s=tau_s, J=J, g=g, E_r=E_r, q=q, u_init=u_init)

# define outputs
outputs = {'s': {'idx': np.arange(2*N, 3*N), 'avg': True}, 'v': {'idx': np.arange(0, N), 'avg': True}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, parallel=True, fastmath=True)

# plot results
fig, ax = plt.subplots(nrows=2, figsize=(12, 6))
ax[0].plot(np.mean(res["s"], axis=1))
ax[0].set_ylabel(r'$s(t)$')
ax[1].plot(np.mean(res["v"], axis=1))
ax[1].set_ylabel(r'$v(t)$')
ax[1].set_xlabel('time')
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/fs_rnn.p", "wb"))
