import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(2)
import numpy as np
from pyrecu import RNN
from pyrecu.neural_models import ik
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

N = 1000

# model parameters
C = 15.0
k = 1.0
v_r = -80.0
v_t = -30.0
v_spike = 1000.0
v_reset = -1000.0
Delta = 2.0
d = 90.0
a = 0.01
b = -20.0
tau_s = 4.0
J = 5.0
g = 5.0
q = 0.0
E_r = -60.0
W = np.load('config/msn_conn.npy')

# define lorentzian of etas
spike_thresholds = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2*v_t-v_r)

# define inputs
T = 100.0
cutoff = 90.0
start = 30.0
stop = 60.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 300.0
# inp[int(start/dt):int(stop/dt)] += 50.0

# run the model
###############

# initialize model
u_init = np.zeros((3*N,))
u_init[:N] += v_r
model = RNN(N, 3*N, ik, W=W, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset, d=d, a=a, b=b,
            tau_s=tau_s, J=J, g=g, E_r=E_r, q=q, u_init=u_init)

# define outputs
outputs = {'s': {'idx': np.arange(2*N, 3*N), 'avg': True}, 'v': {'idx': np.arange(0, N), 'avg': True}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, fastmath=True, method='euler')

# plot results
fig, ax = plt.subplots(nrows=2, figsize=(12, 6))
ax[0].plot(res["s"])
ax[0].set_ylabel(r'$s(t)$')
ax[1].plot(res["v"])
ax[1].set_ylabel(r'$v(t)$')
ax[1].set_xlabel('time')
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'results': res}, open("results/spn_rnn.p", "wb"))
