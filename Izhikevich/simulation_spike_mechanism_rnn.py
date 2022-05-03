import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu import RNN
import matplotlib.pyplot as plt
import pickle
from scipy.stats import cauchy
plt.rcParams['backend'] = 'TkAgg'


def get_fr(inp: np.ndarray, k: float, C: float, v_reset: float, v_spike: float, v_r: float, v_t: float):
    fr = np.zeros_like(inp)
    alpha = v_r+v_t
    mu = 4*(v_r*v_t + inp/k) - alpha**2
    idx = mu > 0
    mu_sqrt = np.sqrt(mu[idx])
    fr[idx] = k*mu_sqrt/(2*C*(np.arctan((2*v_spike-alpha)/mu_sqrt) - np.arctan((2*v_reset-alpha)/mu_sqrt)))
    return fr


def correct_input(inp: np.ndarray, k: float, C: float, v_reset: float, v_spike: float, v_r: float, v_t: float):
    fr = get_fr(inp, k, C, v_reset, v_spike, v_r, v_t)
    inp[fr > 0] = (fr[fr > 0]*C*2*np.pi/k)**2
    return inp


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def ik_ata(y: np.ndarray, N: int, inp: np.ndarray, v_r: float, v_t: np.ndarray, k: float, E_r: float, C: float,
           J: float, g: float, tau_s: float, v_spike: float, v_reset: float, inp_x: np.ndarray, inp_y: np.ndarray,
           dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities."""

    # extract state variables from u
    v, s = y[:N], y[N]

    # calculate network input
    spikes = v >= v_spike
    rates = np.mean(spikes / dt)

    # calculate vector field of the system
    I = (k*v_r*v_t + inp + g*s*(E_r - v))/C
    I_corrected = np.interp(I, inp_x, inp_y)
    dv = (k*(v**2 - (v_r+v_t)*v))/C + I_corrected
    ds = J*rates - s/tau_s

    # update state variables
    v_new = v + dt * dv
    s_new = s + dt * ds

    # reset membrane potential and apply spike frequency adaptation
    v_new[spikes] = v_reset

    # store updated state variables
    y[:N] = v_new
    y[N] = s_new
    y[N+1] = rates

    return y


# define parameters
###################

# model parameters
N = 10000
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
v_spike = 50.0  # unit: mV
v_reset = -100.0  # unit: mV
Delta = 1.0  # unit: mV
tau_s = 6.0
J = 1.0
g = 15.0
E_r = 0.0

# define lorentzian of etas
spike_thresholds = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=0.0)

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 100.0
inp[int(1000/dt):int(2000/dt)] += 100.0

# define input ranges and corrections
inp_y = np.linspace(0.001, 200.0, 1000)
inp_x = correct_input(inp_y, k=k, C=C, v_spike=v_spike, v_reset=v_reset, v_r=v_r, v_t=v_t)

# run the model
###############

# initialize model
u_init = np.zeros((N+2,))
u_init[:N] -= 60.0
model = RNN(N, N+2, ik_ata, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset,
            tau_s=tau_s, J=J, g=g, E_r=E_r, u_init=u_init, inp_x=inp_x, inp_y=inp_y)

# define outputs
outputs = {'s': {'idx': np.asarray([N]), 'avg': False}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, parallel=True, fastmath=True)

# plot results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(np.mean(res["s"], axis=1))
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel('time')
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/spike_mech_rnn.p", "wb"))
