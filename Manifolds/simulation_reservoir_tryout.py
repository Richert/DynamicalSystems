from pyrecu import RNN, random_connectivity
from pyrates import CircuitTemplate
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d
from typing import Union, Callable


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
p = 0.05
W = random_connectivity(N, p)

# input matrix
m = 2
W_in = np.random.randn(N, m)

# simulation parameters
cutoff = 1000.0
T = 10000.0 + cutoff
dt = 1e-2
dts = 1e-1

# initial state
u_init = np.zeros((2*N+1,))
u_init[:N] -= v_r

# define outputs
outputs = {'v': {'idx': np.arange(0, N), 'avg': False}}

# STEP 3: Evaluate classification performance of RNN
####################################################

data = dict()
data["score"] = np.zeros((n_iv,))
data["wta_score"] = np.zeros_like(data["score"])
data["r_qif"] = np.zeros((n_iv, t))
data["v_qif"] = np.zeros_like(data["r_qif"])
data["r_mf"] = np.zeros_like(data["r_qif"])
data["v_mf"] = np.zeros_like(data["r_qif"])
data["Z_qif"] = np.zeros_like(data["r_qif"])
data["Z_mf"] = np.zeros_like(data["r_qif"])
data["iv"] = ivs
data["iv_name"] = iv_name

# simulation loop for n_etas
for j in range(n_iv):

    iv = ivs[j]
    print(f'Performing simulations for {iv_name} = {iv} ...')

    # setup QIF RNN
    qif_rnn = QIFExpAddSyns(C, eta, iv, Delta=Delta, alpha=alpha, tau_s=tau_s, tau_a=tau_a, tau=1.0)

    # perform simulation
    results = qif_rnn.run(T, dt, dts, inp=inp, W_in=W_in, cutoff=cutoff, outputs=(np.arange(0, N), np.arange(3*N, 4*N)),
                          verbose=False)
    v_qif = np.mean(results[0], axis=1)
    r_qif = np.mean(results[1], axis=1)
    X = results[1]

    # prepare training data
    buffer_val = 0
    for i in range(X.shape[1]):
        X[:, i] = gaussian_filter1d(X[:, i], 0.1/dts, mode='constant', cval=buffer_val)
    r_qif2 = np.mean(X, axis=1)

    # split into test and training data
    split = int(np.round(X.shape[0]*0.75, decimals=0))
    X_train = X[:split, :]
    y_train = y[:split, :]
    X_test = X[split:, :]
    y_test = y[split:, :]

    # train RNN
    key, scores, coefs = qif_rnn.ridge_fit(X=X_train, y=y_train, alpha=ridge_alpha, k=0, fit_intercept=False,
                                           copy_X=True, solver='lsqr', readout_key=f'qif_m{M}', verbose=False)

    # calculate classification score on test data
    score, y_predict = qif_rnn.test(X=X_test, y=y_test, readout_key=key)

    # Winner takes it all classification
    wta_pred = y_predict.argmax(axis=1)
    wta_target = y_test.argmax(axis=1)
    wta_score = np.mean(wta_pred == wta_target)

    # simulate mean-field dynamics
    qif_mf = mQIFExpAddSynsRNN(C_m, eta, iv, Delta=Delta, alpha=alpha, tau_a=tau_a, tau_s=tau_s, tau=1.0)
    results = qif_mf.run(T, dt, dts, cutoff=cutoff, outputs=([0], [1]), inp=inp_mf, W_in=np.ones((1, 1)))
    v_mf = np.squeeze(results[0])
    r_mf = np.squeeze(results[1])

    # calculate Kuramoto order parameter Z for QIF network and mean-field model
    Z_qif = kuramoto_order_parameter(r_qif, v_qif)
    Z_mf = kuramoto_order_parameter(r_mf, v_mf)

    print(f"Finished. Results: WTA = {wta_score}, mean(Z) = {np.mean(Z_qif)}.")

    # store data
    data["score"][j] = score
    data["wta_score"][j] = wta_score
    data["r_qif"][j, :] = r_qif2
    data["v_qif"][j, :] = v_qif
    data["r_mf"][j, :] = r_mf
    data["v_mf"][j, :] = v_mf
    data["Z_qif"][j, :] = Z_qif
    data["Z_mf"][j, :] = Z_mf

data["T"] = T
pickle.dump(data, open('data/qif_rc_multichannel_results.pkl', 'wb'))
