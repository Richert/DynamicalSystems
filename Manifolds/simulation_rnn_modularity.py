import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu.neural_models import ik_spike_reset
from pyrecu import RNN, random_connectivity, modularity, sort_via_modules
from typing import Union, Callable
import matplotlib.pyplot as plt
import pickle


# function definitions
######################

def ik(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple, eta: float,
       v_r: float, v_t: float, k: float, E_r: float, C: float, g: float, tau_s: float, b: float, a: float, d: float,
       W: np.ndarray) -> np.ndarray:
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
p = 0.6
W = random_connectivity(N, p)

# input matrix
m = 1
W_in = np.random.randn(N, m)

# simulation parameters
T = 2500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1

# initial state
u_init = np.zeros((2*N+1,))
u_init[:N] -= v_r

# define outputs
outputs = {'v': {'idx': np.arange(0, N), 'avg': False}}

# define inputs
steps = int(T/dt)
inp = np.zeros((m, steps))
#inp[:, :] = np.random.poisson(10.0, (m, steps))

# collect parameters
func_args = (eta_dist, v_r, v_t, k, E_r, C, g, tau_s, b, a, d, W)
callback_args = (v_spike, v_reset)

# perform simulation
####################

# initialize model
model = RNN(N, 2*N+1, ik, func_args, ik_spike_reset, callback_args, u_init=u_init)

# run simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, cutoff=cutoff, solver='heun', inp=inp, W_in=W_in,
                decorator=nb.njit, fastmath=True)

# calculate modularity
######################

# z-transform membrane potentials
z = np.zeros_like(res['v'])
for idx in range(z.shape[1]):
    z_tmp = res['v'][:, idx]
    z_tmp -= np.mean(z_tmp)
    z_tmp /= np.max(np.abs(z_tmp))
    z[:, idx] = z_tmp

# calculate functional modularity of network
modules, A, nodes = modularity(z.T, threshold=0.1, min_connections=4, min_nodes=4, cross_corr_method='fft',
                               decorator=None)

# re-arrange adjacency matrix according to modules
C = sort_via_modules(A, modules)

# save results
pickle.dump({'A': A, 'C': C, 'modules': modules, 'W': W, 'W_in': W_in, 'etas': eta_dist, 'nodes': nodes},
            open("results/modularity_hetconn.p", "wb"))

# plotting
##########

fig1, ax1 = plt.subplots(ncols=2, figsize=(10, 5))
ax1[0].imshow(A)
ax1[0].set_title('Adjacency')
ax1[1].imshow(C, cmap='nipy_spectral')
ax1[1].set_title('Modules')
plt.show()

# fig2, ax2 = plt.subplots(figsize=(10, 4))
# ax2.plot(z)
plt.show()
