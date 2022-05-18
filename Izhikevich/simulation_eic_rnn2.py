import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu import RNN
import matplotlib.pyplot as plt
import pickle
from scipy.stats import cauchy
plt.rcParams['backend'] = 'TkAgg'


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def ik_ei_ata(y: np.ndarray, N: int, inp: np.ndarray, ve_r: float, vi_r: float, ve_t: np.ndarray, vi_t: np.ndarray,
              ke: float, ki: float, Ce: float, Ci: float, ae: float, ai:float, be: float, bi: float, de: float,
              di:float, ve_spike: float, vi_spike: float, ve_reset: float, vi_reset: float, g_ampa: float,
              g_gaba: float, E_ampa: float, E_gaba: float, tau_ampa: float, tau_gaba: float, k_ee: float, k_ei: float,
              k_ie: float, k_ii: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities, split into an excitatory and an inhibitory population.
     """

    # preparatory calculations
    ##########################

    # extract state variables from u
    m = 2*N
    ve, vi = y[:N], y[N:m]
    ue, ui, se_ampa, se_gaba, si_ampa, si_gaba = y[m], y[m+1], y[m+2], y[m+3], y[m+4], y[m+5]

    # extract inputs
    inp_e, inp_i = inp[0], inp[1]

    # calculate network firing rates
    spikes_e = ve >= ve_spike
    rates_e = np.mean(spikes_e / dt)
    spikes_i = vi >= vi_spike
    rates_i = np.mean(spikes_i / dt)

    # calculate vector field of the system
    ######################################

    dy1 = [
        # excitatory population
        (ke*(ve**2 - (ve_r+ve_t)*ve + ve_r*ve_t) + inp_e + g_ampa*se_ampa*(E_ampa-ve) + g_gaba*se_gaba*(E_gaba-ve) - ue) / Ce,

        # inhibitory population
        (ki * (vi**2 - (vi_r+vi_t)*vi + vi_r*vi_t) + inp_i + g_ampa*si_ampa*(E_ampa-vi) + g_gaba*si_gaba*(E_gaba-vi) - ui) / Ci,
    ]
    dy2 = [
        # recovery variables
        ae * (be * (np.mean(ve) - ve_r) - ue) + de*rates_e,
        ai * (bi * (np.mean(vi) - vi_r) - ui) + di*rates_i,

        # synapses
        k_ee*rates_e - se_ampa / tau_ampa,
        k_ei*rates_i - se_gaba / tau_gaba,
        k_ie*rates_e - si_ampa / tau_ampa,
        k_ii*rates_i - si_gaba / tau_gaba,
    ]

    # update state variables
    ########################

    # euler integration
    for i in nb.prange(len(dy1)):
        y[i*N:(i+1)*N] += dt * dy1[i]
    for j in nb.prange(len(dy2)):
        y[m+j] += dt * dy2[j]

    # reset membrane potential and apply spike frequency adaptation
    y[:N][spikes_e] = ve_reset
    y[N:m][spikes_i] = vi_reset

    return y


# define parameters
###################

N = 10000

# define model parameters
#########################

# RS neuron parameters
Ce = 100.0   # unit: pF
ke = 0.7  # unit: None
ve_r = -60.0  # unit: mV
ve_t = -40.0  # unit: mV
ve_spike = 40.0  # unit: mV
ve_reset = -60.0  # unit: mV
Delta_e = 1.0  # unit: mV
de = 20.0
ae = 0.03
be = -2.0

# IB neuron parameters
Ci = 20.0   # unit: pF
ki = 1.0  # unit: None
vi_r = -55.0  # unit: mV
vi_t = -40.0  # unit: mV
vi_spike = 40.0  # unit: mV
vi_reset = -60.0  # unit: mV
Delta_i = 0.3  # unit: mV
di = 0.0
ai = 0.2
bi = 0.025

# synaptic parameters
g_ampa = 1.0
g_gaba = 1.0
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 6.0
tau_gaba = 8.0
k_ee = 16.0
k_ei = 16.0
k_ie = 4.0
k_ii = 4.0

# define lorentzian of etas
spike_thresholds_e = lorentzian(N, eta=ve_t, delta=Delta_e, lb=ve_r, ub=ve_r-ve_t)
spike_thresholds_i = lorentzian(N, eta=vi_t, delta=Delta_i, lb=vi_r, ub=vi_r-vi_t)

# define inputs
T = 4000.0
cutoff = 1000.0
dt = 1e-3
dts = 1e-1
I_ext = np.zeros((int(T/dt), 2))
I_ext[:, 0] += 50.0
I_ext[:, 1] += 36.0
I_ext[int(2000/dt):int(3000/dt), 1] += 14.0
I_ext[int(2500/dt):int(3000/dt), 1] += 25.0

# run the model
###############

# initialize model
u_init = np.zeros((2*N+6,))
u_init[:N] -= 60.0
u_init[N:2*N] -= 60.0
model = RNN(N, 2*N+6, ik_ei_ata, Ce=Ce, Ci=Ci, ke=ke, ki=ki, ve_r=ve_r, vi_r=vi_r, ve_t=spike_thresholds_e,
            vi_t=spike_thresholds_i, ve_spike=ve_spike, vi_spike=vi_spike, ve_reset=ve_reset, vi_reset=vi_reset,
            de=de, di=di, ae=ae, ai=ai, be=be, bi=bi, g_ampa=g_ampa, g_gaba=g_gaba, E_ampa=E_ampa, E_gaba=E_gaba,
            tau_ampa=tau_ampa, tau_gaba=tau_gaba, k_ee=k_ee, k_ei=k_ei, k_ie=k_ie, k_ii=k_ii, u_init=u_init)

# define outputs
outputs = {'se': {'idx': np.asarray([2*N+2]), 'avg': False}, 'si': {'idx': np.asarray([2*N+3]), 'avg': False}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=I_ext, cutoff=cutoff, parallel=True, fastmath=True)

# plot results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(res["se"]/k_ee)
ax.plot(res["si"]/k_ei)
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel('time')
plt.legend(['RS', 'FS'])
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/eic_rnn_het.p", "wb"))
