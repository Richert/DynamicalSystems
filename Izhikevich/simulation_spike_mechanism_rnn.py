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
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
v_spike = 50.0  # unit: mV
v_reset = -70.0  # unit: mV
Delta = 0.5  # unit: mV
d = 10.0
a = 0.03
b = -2.0
tau_s = 6.0
g = 15.0
J = 1.0
E_r = 0.0

# define lorentzian of etas
spike_thresholds = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=0.0)

# define inputs
T = 4500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 30.0
inp[int(1500/dt):int(2500/dt)] = 50.0
inp[int(2500/dt):int(3500/dt)] = 13.0

# run the model
###############

# initialize model
u_init = np.zeros((2*N+2,))
u_init[:N] -= 60.0
model = RNN(N, 2*N+2, ik_ata, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset, d=d, a=a, b=b,
            tau_s=tau_s, J=J, g=g, E_r=E_r, q=0.0, u_init=u_init)

# define outputs
outputs = {'s': {'idx': np.asarray([2*N]), 'avg': False}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, parallel=True, fastmath=True)

# plot results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(np.mean(res["s"], axis=1))
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel('time')
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/spike_mech_rnn.p", "wb"))
