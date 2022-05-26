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


@nb.njit(fastmath=True)
def correct_input(inp: float, k: float, v_r: float, v_t: float, v_spike: float, v_reset: float, g: float, s: float,
                  E: float, u: float, Delta: float):
    alpha = v_r + v_t + g*s/k
    mu = 4*(v_r*v_t + (inp - u*np.pi**(2/3) + g*s*E)/k) - alpha**2
    if mu > 0:
        mu_sqrt = np.sqrt(mu)
        inp_c = np.pi**2*k*mu/(4*(np.arctan((2*v_spike-alpha)/mu_sqrt) - np.arctan((2*v_reset-alpha)/mu_sqrt))**2)
        inp_c += k*alpha**2/4 + u*np.pi**(2/3)
        inp_c -= k*v_r*v_t + g*s*E
    else:
        inp_c = inp
    return inp_c


def ik_run(y, N, inp, v_t, v_r, k, J, g, Delta, C, v_z, v_p, E_r, b, a, d, tau_s, dt):

    r = y[0]
    v = y[1]
    u = y[2]
    s = y[3]
    r_in = J*r
    I_ext = correct_input(inp, k, v_r, v_t, v_p, v_z, g, s, E_r, u, Delta)

    dr = (r * (-g * s + k * (2.0 * v - v_r - v_t)) + Delta * k ** 2 * (v - v_r) / (np.pi * C)) / C
    dv = (-np.pi * C * r * (np.pi * C * r / k + Delta) + I_ext + g * s * (
                E_r - v) + k * v * (v - v_r - v_t) + k * v_r * v_t - u) / C
    du = a * (b * (v - v_r) - u) + d * r
    ds = r_in - s / tau_s

    y[0] += dt * dr
    y[1] += dt * dv
    y[2] += dt * du
    y[3] += dt * ds

    return y


def ik_ata(y: np.ndarray, N: int, inp: np.ndarray, v_r: float, v_t: np.ndarray, k: float, E_r: float, C: float,
           J: float, g: float, tau_s: float, b: float, a: float, d: float, v_spike: float, v_reset: float,
           dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities."""

    # extract state variables from u
    v, u, s = y[:N], y[N], y[N+1]

    # calculate network input
    spikes = v >= v_spike
    rates = np.mean(spikes / dt)

    # calculate vector field of the system
    dv = (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + g*s*(E_r - v) - u)/C
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
v_spike = 2000.0  # unit: mV
v_reset = -2000.0  # unit: mV
d = 20.0
a = 0.03
b = -2.0
tau_s = 6.0
J = 1.0
g = 15.0
E_r = 0.0

# SNN-specific variables
N = 10000
u_init = np.zeros((N+3,))
u_init[:N] -= 60.0
u_mf = np.zeros((4,))
u_mf[1] = -60.0
outputs = {'s': {'idx': np.asarray([N+1]), 'avg': False}}
outputs_mf = {'s': {'idx': np.asarray([3]), 'avg': False}}

# define inputs
T = 5000.0
cutoff = 2000.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 60.0

#######################################################
# calculate FRE vs SNN differences for various deltas #
#######################################################

n = 100
deltas = np.linspace(0.01, 10.0, num=n)
signals = {'fre': [], 'snn': []}
for Delta in deltas:

    # fre simulation
    ################

    # initialize model
    model = RNN(4, 4, ik_run, C=C, k=k, Delta=Delta, v_r=v_r, v_t=v_t, v_p=v_spike, v_z=v_reset, d=d, a=a,
                b=b, tau_s=tau_s, J=J, g=g, E_r=E_r, u_init=u_mf)

    # perform simulation
    fre = model.run(T=T, dt=dt, dts=dts, outputs=outputs_mf, inp=inp, cutoff=cutoff, fastmath=True)

    # snn simulation
    ################

    # define lorentzian of spike thresholds
    spike_thresholds = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=v_r-v_t)

    # initialize model
    model = RNN(N, N+3, ik_ata, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset, d=d, a=a,
                b=b, tau_s=tau_s, J=J, g=g, E_r=E_r, u_init=u_init)

    # perform simulation
    snn = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, parallel=True, fastmath=True)

    # save signals
    signals['fre'].append(fre)
    signals['snn'].append(snn)

    print(fr"$\Delta = {Delta}$")
    print(f"Diff: {np.mean(fre['s'].squeeze()-snn['s'].squeeze())}")

    # # plot results
    # fig, ax = plt.subplots(figsize=(12, 4))
    # ax.plot(snn["s"])
    # ax.plot(fre["s"])
    # plt.legend(['SNN', 'MF'])
    # plt.show()

# save results
pickle.dump({'results': signals}, open("results/rs_lorentzian.p", "wb"))
