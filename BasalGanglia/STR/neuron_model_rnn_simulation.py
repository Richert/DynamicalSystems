import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(2)
import numpy as np
from rectipy import Network
import matplotlib.pyplot as plt
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
C = 15.0
k = 1.0
v_r = -80.0
v_t = -30.0
v_spike = 15.0
v_reset = -90.0
Delta = 2.0
d = 90.0
a = 0.01
b = -20.0
tau_s = 4.0
g = 5.0
E_r = -60.0
W = np.load('config/msn_conn.npy')

# define lorentzian of spike thresholds
spike_thresholds = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2*v_t-v_r)

# define inputs
T = 100.0
cutoff = 0.0
start = 30.0
stop = 60.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 300.0
# inp[int(start/dt):int(stop/dt)] += 50.0

# collect parameters
func_args = (v_r, spike_thresholds, k, E_r, C, g, tau_s, b, a, q)
callback_args = (v_spike, v_reset, J, d)

# run the model
###############

# initialize model
u_init = np.zeros((2*N+1,))
u_init[:N] += -60.0
model = Network.from_yaml()

# define outputs
outputs = {'s': {'idx': np.asarray([2*N]), 'avg': False}, 'v': {'idx': np.arange(0, N), 'avg': True}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, solver='euler', decorator=nb.njit,
                fastmath=True, parallel=True)

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
