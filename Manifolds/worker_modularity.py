import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu.neural_models import ik_spike_reset
from pyrecu import RNN, random_connectivity, modularity
from typing import Union, Callable
import pickle
import sys
from scipy.stats import cauchy


# function definitions
######################

def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


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

cond = 10 #int(sys.argv[1])
deltas = np.arange(0.2, 4.0, 0.2)

# model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = deltas[cond]
eta = 40.0
a = 0.03
b = -2.0
d = 10.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0
N = 1000
p = 0.1

# simulation parameters
T_cutoff = 1000.0
T_compare = 2000.0
T = T_cutoff + T_compare
dt = 1e-2
dts = 2e-1

# initial state
u_init = np.zeros((2*N+1,))
u_init[:N] -= v_r

# define inputs
m = 2
in_var = 10.0
inp = np.zeros((m, int(T/dt)))

# define outputs
outputs = {'v': {'idx': np.arange(0, N), 'avg': False}}

# collect parameters
callback_args = (v_spike, v_reset)

#######################################################
# perform simulations for different background inputs #
#######################################################

n_reps = 10
results = {'v': [], 'W': [], 'W_in': [], 'p': p, 'inp': inp, 'predictions': [], 'scores': [], 'modules': [],
           'adjacency': [], 'nodes': []}
for _ in range(n_reps):

    # simulate signal
    #################

    # theta distribution
    theta_dist = lorentzian(N, v_t, Delta, v_r, 2 * v_t - v_r)

    # sample random connectivity matrix
    W = random_connectivity(N, p)

    # sample random input weights
    W_in = np.random.randn(N, m) * in_var
    for i in range(m):
        W_in[:, i] -= np.sum(W_in[:, i])

    # initialize model
    func_args = (eta, v_r, theta_dist, k, E_r, C, g, tau_s, b, a, d)
    model = RNN(N, 2*N+1, ik, func_args + (W,), ik_spike_reset, callback_args, u_init=u_init)

    # run simulation
    res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, cutoff=T_cutoff, solver='heun', decorator=nb.njit,
                    fastmath=True, inp=inp, W_in=W_in)

    # modularity calculation
    ########################

    # z-transform membrane potentials
    comp_signal = res['v']
    z = np.zeros_like(comp_signal)
    for idx in range(z.shape[1]):
        z_tmp = comp_signal[:, idx]
        z_tmp -= np.mean(z_tmp)
        z_tmp /= np.max(np.abs(z_tmp))
        z[:, idx] = z_tmp

    # calculate functional modularity of network
    modules, A, nodes = modularity(z.T, threshold=0.1, min_connections=5, min_nodes=5, cross_corr_method='fft',
                                   decorator=None)

    # save results
    ##############

    results['v'].append(comp_signal)
    results['W'].append(W)
    results['W_in'].append(W_in)
    results['modules'].append(modules)
    results['adjacency'].append(A)
    results['nodes'].append(nodes)

    # testing stuff (comment out for cluster computations)
    ######################################################

    # printing
    print(f'Number of modules: {len(modules)}')

    # plotting
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for neuron in [300, 500, 700]:
        ax.plot(z)
    plt.show()

# save results
pickle.dump(results, open(f"/projects/p31302/richard/results/rnn_{cond}.p", "wb"))
