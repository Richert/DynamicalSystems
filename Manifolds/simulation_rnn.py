import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu.neural_models import ik_spike_reset
from pyrecu import RNN, random_connectivity
from typing import Union, Callable
import matplotlib.pyplot as plt
import pickle


# function definitions
######################

def ik(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple, eta: float,
       v_r: float, v_t: np.ndarray, k: float, E_r: float, C: float, g: float, tau_s: float, b: float, a: float,
       d: float, W: np.ndarray) -> np.ndarray:
    """Calculates right-hand side update of a network of coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities."""

    dy = np.zeros_like(y)

    # extract state variables from u
    m = 2*N
    v, u, s = y[:N], y[N:m], y[m:]

    # retrieve extrinsic input at time t
    inp = infunc(t, *inargs)

    # calculate state vector updates
    dy[:N] = (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + g*s*(E_r - v) + eta - u)/C
    dy[N:m] = a*(b*(v-v_r) - u) + d*rates
    dy[m:] = -s/tau_s + rates @ W

    return dy


# parameter definition
######################

# model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 2.0
a = 0.03
b = -2.0
d = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# eta distribution
N = 1000
eta_dist = Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# simulation parameters
T = 1300.0
cutoff = 300.0
dt = 1e-2
dts = 1e-1

# initial state
u_init = np.zeros((3*N,))
u_init[:N] -= v_r

# define outputs
outputs = {'v': {'idx': np.arange(0, N), 'avg': True}, 'u': {'idx': np.arange(N, 2*N), 'avg': True},
           's': {'idx': np.arange(2*N, 3*N), 'avg': False}}

# collect parameters
func_args = (v_r, v_t, k, E_r, C, g, tau_s, b, a, d)
callback_args = (v_spike, v_reset)

# perform simulations for different background inputs
#####################################################

etas = np.arange(30, 80, step=2)
ps = np.arange(1.0, 0.01, step=-0.05)
results = {'results': [], 'etas': [], 'ps': []}
for p in ps:

    # generate connectivit matrix
    W = random_connectivity(N, p)

    for eta in etas:

        # initialize model
        model = RNN(N, 3*N, ik, (eta + eta_dist,) + func_args + (W,), ik_spike_reset, callback_args, u_init=u_init)

        # run simulation
        res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, cutoff=cutoff, solver='midpoint', decorator=nb.njit)

        results['results'].append(res)
        results['etas'].append(eta)
        results['ps'].append(p)

        # plot results
        # fig, ax = plt.subplots(figsize=(12, 4))
        # ax.plot(np.mean(res["v"], axis=1))
        # ax.set_xlabel('time')
        # ax.set_ylabel(r'$v(t)$')
        # plt.tight_layout()
        # plt.show()

# save results
pickle.dump(results, open("results/ik_rnn.p", "wb"))
