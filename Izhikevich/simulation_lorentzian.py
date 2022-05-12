from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
from scipy.stats import cauchy
from pyrecu import RNN


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def ik_ata(y: np.ndarray, N: int, inp: np.ndarray, v_r: float, v_t: np.ndarray, k: float, E_r: float, C: float,
           J: float, g: float, tau_s: float, b: float, a: float, d: float, v_spike: float, v_reset: float,
           q: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities."""

    # extract state variables from u
    v, u, s = y[:N], y[N], y[N+1]

    # calculate network input
    spikes = v >= v_spike
    rates = np.mean(spikes / dt)

    # calculate vector field of the system
    dv = (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + g*s*(E_r - v) + q*(np.mean(v)-v) - u)/C
    du = a*(b*(np.mean(v)-v_r) - u) + d*rates
    ds = J*rates - s/tau_s

    # update state variables
    v_new = v + dt * dv
    u_new = u + dt * du
    s_new = s + dt * ds

    # reset membrane potential and apply spike frequency adaptation
    v_new[spikes] = v_reset

    # store updated state variables
    y[:N] = v_new
    y[N] = u_new
    y[N+1] = s_new
    y[N+2] = rates

    return y


# define parameters
###################

# model parameters
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
v_spike = 1000.0  # unit: mV
v_reset = -1000.0  # unit: mV
d = 20.0
a = 0.03
b = -2.0
tau_s = 6.0
J = 1.0
g = 15.0
q = 0.0
E_r = 0.0

# SNN-specific variables
N = 10000
u_init = np.zeros((N+3,))
u_init[:N] -= 60.0
outputs = {'v': {'idx': np.arange(0, N), 'avg': True}}

# define inputs
T = 5500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 60.0

#######################################################
# calculate FRE vs SNN differences for various deltas #
#######################################################

n = 100
deltas = np.linspace(0.1, 10.0, num=n)
signals = {'fre': [], 'snn': []}
for Delta in deltas:

    # fre simulation
    ################

    # initialize model
    ik = CircuitTemplate.from_yaml("config/ik/ik")

    # update parameters
    ik.update_var(node_vars={'p/ik_op/C': C, 'p/ik_op/k': k, 'p/ik_op/v_r': v_r, 'p/ik_op/v_t': v_t, 'p/ik_op/v_p': v_spike,
                             'p/ik_op/v_z': -v_reset, 'p/ik_op/Delta': Delta, 'p/ik_op/d': d, 'p/ik_op/a': a,
                             'p/ik_op/b': b, 'p/ik_op/tau_s': tau_s, 'p/ik_op/g': g, 'p/ik_op/q': q, 'p/ik_op/E_r': E_r})

    # run simulation
    fre = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
                 outputs={'v': 'p/ik_op/v'}, inputs={'p/ik_op/I_ext': inp},
                 decorator=nb.njit, fastmath=True, clear=True)

    # snn simulation
    ################

    # define lorentzian of spike thresholds
    spike_thresholds = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=v_r-v_t)

    # initialize model
    model = RNN(N, N+3, ik_ata, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset, d=d, a=a,
                b=b, tau_s=tau_s, J=J, g=g, E_r=E_r, q=q, u_init=u_init)

    # perform simulation
    snn = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, parallel=True, fastmath=True)

    # save signals
    signals['fre'].append(fre)
    signals['snn'].append(snn)

    # # plot results
    # fig, ax = plt.subplots(figsize=(12, 4))
    # ax.plot(fre.index, snn["v"])
    # ax.plot(fre["v"])
    # ax.set_ylabel(r'$v(t)$')
    # ax.set_xlabel("time (ms)")
    # plt.legend(['SNN', 'MF'])
    # plt.show()

# save results
pickle.dump({'results': signals}, open("results/rs_lorentzian.p", "wb"))
