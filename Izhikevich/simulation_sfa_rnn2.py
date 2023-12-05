import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu import RNN
from pyrecu.neural_models import ik_spike_reset, ik_ata
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

# model parameters
N = 5000
C = 200.0   # unit: pF
k = 3.0  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
v_spike = 1000.0  # unit: mV
v_reset = -1000.0  # unit: mV
Delta = 1.0  # unit: mV
d = 80.0
a = 0.005
b = -10.0
tau_s = 6.0
J = 1.0
g = 20.0
E_r = 0.0

# define lorentzian of etas
spike_thresholds = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=v_r-v_t)

# define inputs
T = 5500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 40.0
inp[int(1000/dt):int(5000/dt)] += np.linspace(0.0, 500.0, num=int(4000/dt))
#inp[int(3000/dt):int(5000/dt)] += np.linspace(30.0, 0.0, num=int(2000/dt))

# run the model
###############

# initialize model
u_init = np.zeros((2*N+1,))
u_init[:N] -= 60.0
run_args = (v_r, spike_thresholds, k, E_r, C, g, tau_s, b, a, d, 0.0, J)
spike_args = (v_spike, v_reset)
model = RNN(N, 2*N+2, ik_ata, evolution_args=run_args, callback_func=ik_spike_reset, callback_args=spike_args,
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
# pickle.dump({'results': res}, open("results/sfa_rnn2_high.p", "wb"))
