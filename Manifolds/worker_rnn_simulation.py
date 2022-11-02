import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu.neural_models import ik_spike_reset
from pyrecu import RNN, random_connectivity, ReadoutTraining, modularity
from typing import Union, Callable
import pickle
import sys
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


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
eta = 40.0
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
cond = 1 #int(sys.argv[1])
p = ps[cond]

# simulation parameters
T_cutoff = 1000.0
T_compare = 2000.0
T_epoch = 1000.0
n_train_epochs = 1
n_test_epochs = 1
T = T_cutoff + T_compare + T_epoch*(n_train_epochs + n_test_epochs)
dt = 1e-2
dts = 2e-1

# initial state
u_init = np.zeros((2*N+1,))
u_init[:N] -= v_r

# define inputs
m = 2
in_var = 10.0
inp = np.zeros((m, int(T/dt)))
targets = np.zeros((int((T-T_cutoff-T_compare)/dts), m))
in_start = int((T_compare + T_cutoff)/dt)
in_steps = int(T_epoch / dt)
target_steps = int(T_epoch/dts)
target_start = int(T_compare/dts)
split = int(T_epoch*n_train_epochs/dts)
time1 = np.linspace(0, T_epoch * (n_train_epochs + n_test_epochs), num=in_steps)
time2 = np.linspace(0, T_epoch * (n_train_epochs + n_test_epochs), num=target_steps)
s1 = np.sin(2.0*np.pi*time1*0.004)
s2 = np.sin(2.0*np.pi*time1*0.002)
s3 = np.sin(2.0*np.pi*time2*0.004)
s4 = np.sin(2.0*np.pi*time2*0.002)
for i in range(n_train_epochs + n_test_epochs):
    inp[0, in_start + i * in_steps:in_start + (i + 1) * in_steps] = s1
    inp[1, in_start + i * in_steps:in_start + (i + 1) * in_steps] = s2
    targets[i * target_steps:(i + 1) * target_steps, 0] = s3 * s4
    targets[i * target_steps:(i + 1) * target_steps, 1] = s3 + s4

# define outputs
outputs = {'v': {'idx': np.arange(0, N), 'avg': False}}

# collect parameters
func_args = (eta_dist, v_r, v_t, k, E_r, C, g, tau_s, b, a, d)
callback_args = (v_spike, v_reset)

#######################################################
# perform simulations for different background inputs #
#######################################################

n_reps = 10
results = {'v': [], 'W': [], 'W_in': [], 'p': p, 'inp': inp, 'targets': targets, 'predictions': [], 'scores': [],
           'modules': [], 'adjacency': [], 'nodes': []}
for _ in range(n_reps):

    # simulate signal
    #################

    # sample random connectivity matrix
    W = random_connectivity(N, p)

    # sample random input weights
    W_in = np.random.randn(N, m) * in_var
    for i in range(m):
        W_in[:, i] -= np.sum(W_in[:, i])

    # initialize model
    model = RNN(N, 2*N+1, ik, func_args + (W,), ik_spike_reset, callback_args, u_init=u_init)

    # run simulation
    res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, cutoff=T_cutoff, solver='heun', decorator=nb.njit,
                    fastmath=True, inp=inp, W_in=W_in)

    # reservoir computing
    #####################

    # find spikes
    #spikes = []
    res_signal = res['v'][target_start:, :]
    X = np.zeros_like(res_signal)
    for i in range(X.shape[1]):
        peaks, _ = find_peaks(res_signal[:, i])
        #spikes.append(peaks)
        X[np.squeeze(peaks), i] = 1.0

    # prepare training data
    buffer_val = 0
    for i in range(X.shape[1]):
        X[:, i] = gaussian_filter1d(X[:, i], 2.0/dts, mode='constant', cval=buffer_val)

    # split into test and training data
    split = int(np.round(X.shape[0] * 0.75, decimals=0))
    X_train = X[:split, :]
    y_train = targets[:split, :]
    X_test = X[split:, :]
    y_test = targets[split:, :]

    # train RNN
    rnn = ReadoutTraining()
    key, scores, coefs = rnn.ridge_fit(X=X_train, y=y_train, alpha=5e-3, k=0, fit_intercept=False,
                                       copy_X=True, solver='lsqr', readout_key=f'test', verbose=False)

    # calculate classification score on test data
    score, y_predict = rnn.test(X=X_test, y=y_test, readout_key=key)

    # modularity calculation
    ########################

    # z-transform membrane potentials
    comp_signal = res['v'][:target_start, :]
    z = np.zeros_like(comp_signal)
    for idx in range(z.shape[1]):
        z_tmp = comp_signal[:, idx]
        z_tmp -= np.mean(z_tmp)
        z_tmp /= np.max(np.abs(z_tmp))
        z[:, idx] = z_tmp

    # calculate functional modularity of network
    modules, A, nodes = modularity(z.T, threshold=0.1, min_connections=4, min_nodes=4, cross_corr_method='fft',
                                   decorator=None)

    # save results
    ##############

    results['v'].append(comp_signal)
    results['W'].append(W)
    results['W_in'].append(W_in)
    results['predictions'].append(y_predict)
    results['scores'].append(score)
    #results['spikes'].append(spikes)
    results['modules'].append(modules)
    results['adjacency'].append(A)
    results['nodes'].append(nodes)

    # testing stuff (comment out for cluster computations)
    ######################################################

    # printing
    print(f'Test score: {score}')
    print(f'Number of modules: {len(modules)}')

    # plotting
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=3, figsize=(10, 6))
    axes[0].plot(X)
    axes[1].plot(y_predict[:, 0])
    axes[1].plot(y_test[:, 0])
    axes[2].plot(y_predict[:, 1])
    axes[2].plot(y_test[:, 1])
    plt.show()

# save results
pickle.dump(results, open(f"/projects/p31302/richard/results/rnn_{cond}.p", "wb"))
