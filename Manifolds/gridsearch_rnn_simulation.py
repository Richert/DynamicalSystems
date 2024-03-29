import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu.neural_models import ik_spike_reset
from pyrecu import RNN
from typing import Union, Callable
import pickle


# function definitions
######################

def ik(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple, eta: float,
       v_r: float, v_t: np.ndarray, k: float, E_r: float, C: float, g: float, tau_s: float, b: float, a: float,
       d: float) -> np.ndarray:
    """Calculates right-hand side update of a network of coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities."""

    dy = np.zeros_like(y)

    # extract state variables from u
    v, u, s = y[:N], y[N], y[N+1]

    # retrieve extrinsic input at time t
    inp = infunc(t, *inargs)

    # calculate average firing rate
    r = np.mean(rates)

    # calculate state vector updates
    dy[:N] = (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + g*s*(E_r - v) + eta - u)/C
    dy[N] = a*(b*(np.mean(v)-v_r) - u) + d*r
    dy[N+1] = -s/tau_s + r

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
func_args = (v_r, v_t, k, E_r, C, g, tau_s, b, a, d)
callback_args = (v_spike, v_reset)

# perform simulations for different background inputs
#####################################################

etas = np.asarray([45.0, 60.0, 75.0])
results = {'results': [], 'etas': etas}
for eta in etas:

    # initialize model
    model = RNN(N, 3*N, ik, (eta + eta_dist,) + func_args, ik_spike_reset, callback_args, u_init=u_init)

    # run simulation
    res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, cutoff=cutoff, solver='midpoint', decorator=nb.njit,
                    fastmath=True)

    # store results
    results['results'].append(res)

# save results
pickle.dump(results, open(f"results/rnn_results.p", "wb"))
