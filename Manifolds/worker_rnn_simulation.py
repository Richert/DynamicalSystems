import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu.neural_models import ik_spike_reset
from pyrecu import RNN, random_connectivity
from typing import Union, Callable
import pickle
import sys


# function definitions
######################

def ik(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple, eta: float,
       v_r: float, v_t: np.ndarray, k: float, E_r: float, C: float, g: float, tau_s: float, b: float, a: float,
       d: float, W: np.ndarray) -> np.ndarray:
    """Calculates right-hand side update of a network of coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities."""

    dy = np.zeros_like(y)

    # extract state variables from u
    v, u, s = y[:N], y[N], y[N+1:]

    # retrieve extrinsic input at time t
    inp = infunc(t, *inargs)

    # calculate state vector updates
    dy[:N] = (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + g*s*(E_r - v) + eta - u)/C
    dy[N] = a*(b*(np.mean(v)-v_r) - u) + d*np.mean(rates)
    dy[N+1:] = -s/tau_s + W @ rates

    return dy


# parameter definition
######################

# model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 2.0
eta = 45.0
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
eta_dist = eta + Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# connectivity matrix
ps = 1 / 2**(np.asarray([0, 1, 2, 3, 4, 5, 6]))
cond = int(sys.argv[1])
p = ps[cond]
W = random_connectivity(N, p)

# simulation parameters
T = 2500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1

# initial state
u_init = np.zeros((2*N+1,))
u_init[:N] -= v_r

# define outputs
outputs = {'v': {'idx': np.arange(0, N), 'avg': True}}

# collect parameters
func_args = (v_r, v_t, k, E_r, C, g, tau_s, b, a, d, W)
callback_args = (v_spike, v_reset)

# perform simulations for different background inputs
#####################################################

in_var = np.asarray([0.0, 0.025, 0.5, 1.0, 2.0, 4.0, 8.0])
results = {'results': [], 'in_var': in_var, 'etas': [], 'p': p, 'W': W}
for v in in_var:

    # draw random input weights
    eta_in = np.random.randn(N) * v
    eta_in -= np.mean(eta_in)

    # initialize model
    model = RNN(N, 3*N, ik, (eta_in + eta_dist,) + func_args, ik_spike_reset, callback_args, u_init=u_init)

    # run simulation
    res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, cutoff=cutoff, solver='midpoint', decorator=nb.njit,
                    fastmath=True)

    # store results
    results['results'].append(res)
    results['etas'].append(eta_in + eta_dist - eta)

# save results
pickle.dump(results, open(f"results/rnn_{cond}.p", "wb"))
