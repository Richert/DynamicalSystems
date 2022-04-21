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


def eii_run(y: np.ndarray, N: int, inp: np.ndarray, ve_r: float, vi_r: float, vi2_r: float, ve_t: np.ndarray,
            vi_t: np.ndarray, vi2_t: np.ndarray, ke: float, ki: float, ki2: float, Ce: float, Ci: float, Ci2: float,
            ae: float, ai:float, ai2:float, be: float, bi: float, bi2: float, de: float, di:float, di2:float,
            ve_spike: float, vi_spike: float, vi2_spike: float, ve_reset: float, vi_reset: float, vi2_reset: float,
            g_ampa: float, g_gaba: float, E_ampa: float, E_gaba: float, tau_ampa: float, tau_gaba: float, k_e: float,
            k_i: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities, split into an excitatory and two inhibitory populations.
     """

    # preparatory calculations
    ##########################

    # extract state variables from u
    m = 6*N
    ve, ue, vi, ui, vi2, ui2 = y[:N], y[N:2*N], y[2*N:3*N], y[3*N:4*N], y[4*N:5*N], y[5*N:m]
    se, si, si2, re, ri, ri2 = y[m], y[m+1], y[m+2], y[m+3], y[m+4], y[m+5]

    # extract inputs
    inp_e, inp_i, inp_i2 = inp[0], inp[1], inp[2]

    # calculate network firing rates
    spikes_e = ve >= ve_spike
    rates_e = np.mean(spikes_e / dt)
    spikes_i = vi >= vi_spike
    rates_i = np.mean(spikes_i / dt)
    spikes_i2 = vi2 >= vi2_spike
    rates_i2 = np.mean(spikes_i2 / dt)

    # calculate vector field of the system
    ######################################

    # rs population
    d_ve = (ke*(ve**2 - (ve_r+ve_t)*ve + ve_r*ve_t) + inp_e + k_e*g_ampa*se*(E_ampa-ve) + k_e*g_gaba*(si+si2)*(E_gaba-ve) - ue)/Ce
    d_ue = ae*(be*(ve-ve_r) - ue)
    d_se = rates_e - se/tau_ampa

    # fs population
    d_vi = (ki*(vi**2 - (vi_r+vi_t)*vi + vi_r*vi_t) + inp_i + k_i*g_ampa*se*(E_ampa-vi) + k_i*g_gaba*(si+si2)*(E_gaba-vi) - ui) / Ci
    d_ui = ai*(bi*(vi-vi_r) - ui)
    d_si = rates_i - si/tau_gaba

    # lts population
    d_vi2 = (ki2*(vi2**2 - (vi2_r+vi2_t)*vi2 + vi2_r*vi2_t) + inp_i2 + k_i*g_ampa*se*(E_ampa - vi2) + k_i*g_gaba*(si+si2)*(E_gaba-vi2) - ui2) / Ci2
    d_ui2 = ai2*(bi2*(vi2-vi2_r) - ui2)
    d_si2 = rates_i2 - si2/tau_gaba

    # update state variables
    ########################

    # euler integration
    ve_new = ve + dt * d_ve
    ue_new = ue + dt * d_ue
    se_new = se + dt * d_se
    vi_new = vi + dt * d_vi
    ui_new = ui + dt * d_ui
    si_new = si + dt * d_si
    vi2_new = vi2 + dt * d_vi2
    ui2_new = ui2 + dt * d_ui2
    si2_new = si2 + dt * d_si2

    # reset membrane potential and apply spike frequency adaptation
    ve_new[spikes_e] = ve_reset
    ue_new[spikes_e] += de
    vi_new[spikes_i] = vi_reset
    ui_new[spikes_i] += di
    vi2_new[spikes_i2] = vi2_reset
    ui2_new[spikes_i2] += di2

    # store updated state variables
    y[:N] = ve_new
    y[N:2*N] = ue_new
    y[2*N:3*N] = vi_new
    y[3*N:4*N] = ui_new
    y[4*N:5*N] = vi2_new
    y[5*N:m] = ui2_new
    y[m] = se_new
    y[m+1] = si_new
    y[m+2] = si2_new
    y[m+3] = rates_e
    y[m+4] = rates_i
    y[m+5] = rates_i2

    return y


# define parameters
###################

N = 10000

# RS neuron parameters
Ce = 100.0   # unit: pF
ke = 0.7  # unit: None
ve_r = -60.0  # unit: mV
ve_t = -40.0  # unit: mV
ve_spike = 400.0  # unit: mV
ve_reset = -600.0  # unit: mV
Delta_e = 0.5  # unit: mV
de = 20.0
ae = 0.03
be = -2.0

# FS neuron parameters
Ci = 20.0   # unit: pF
ki = 1.0  # unit: None
vi_r = -55.0  # unit: mV
vi_t = -40.0  # unit: mV
vi_spike = 500.0  # unit: mV
vi_reset = -900.0  # unit: mV
Delta_i = 0.3  # unit: mV
di = 0.0
ai = 0.2
bi = 0.025

# LTS neuron parameters
Ci2 = 100.0   # unit: pF
ki2 = 1.0  # unit: None
vi2_r = -56.0  # unit: mV
vi2_t = -42.0  # unit: mV
vi2_spike = 400.0  # unit: mV
vi2_reset = -530.0  # unit: mV
Delta_i2 = 1.5  # unit: mV
di2 = 20.0
ai2 = 0.03
bi2 = 8.0

# synaptic parameters
g_ampa = 1.0
g_gaba = 1.0
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 6.0
tau_gaba = 8.0
k_e = 16.0
k_i = 4.0

# define lorentzian of etas
spike_thresholds_e = lorentzian(N, eta=ve_t, delta=Delta_e, lb=ve_r, ub=0.0)
spike_thresholds_i = lorentzian(N, eta=vi_t, delta=Delta_i, lb=vi_r, ub=0.0)
spike_thresholds_i2 = lorentzian(N, eta=vi2_t, delta=Delta_i2, lb=vi2_r, ub=0.0)

# define inputs
T = 4000.0
cutoff = 1000.0
dt = 1e-3
dts = 1e-1
I_ext = np.zeros((int(T/dt), 3))
I_ext[:, 0] = 50.0
I_ext[:, 1] = 20.0
I_ext[:, 2] = 80.0
I_ext[int(2000/dt):int(3000/dt), 2] += 20.0
I_ext[int(2500/dt):int(3000/dt), 2] += 40.0

# run the model
###############

# initialize model
u_init = np.zeros((6*N+6,))
u_init[:N] -= 45.0
u_init[2*N:3*N] -= 60.0
u_init[4*N:5*N] -= 60.0
model = RNN(N, 6*N, eii_run, Ce=Ce, Ci=Ci, Ci2=Ci2, ke=ke, ki=ki, ki2=ki2, ve_r=ve_r, vi_r=vi_r, vi2_r=vi2_r,
            ve_t=spike_thresholds_e, vi_t=spike_thresholds_i, vi2_t=spike_thresholds_i2, ve_spike=ve_spike,
            vi_spike=vi_spike, vi2_spike=vi2_spike, ve_reset=ve_reset, vi_reset=vi_reset, vi2_reset=vi2_reset,
            de=de, di=di, di2=di2, ae=ae, ai=ai, ai2=ai2, be=be, bi=bi, bi2=bi2, g_ampa=g_ampa, g_gaba=g_gaba,
            E_ampa=E_ampa, E_gaba=E_gaba, tau_ampa=tau_ampa, tau_gaba=tau_gaba, k_e=k_e, k_i=k_i, u_init=u_init)

# define outputs
outputs = {'rs': {'idx': np.asarray([6*N+3]), 'avg': False}, 'fs': {'idx': np.asarray([6*N+4]), 'avg': False},
           'lts': {'idx': np.asarray([6*N+5]), 'avg': False}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=I_ext, cutoff=cutoff, parallel=True, fastmath=True)

# plot results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(res["rs"])
ax.plot(res["fs"])
ax.plot(res["lts"])
ax.set_ylabel(r'$r(t)$')
ax.set_xlabel('time')
plt.legend(['RS', 'FS', 'LTS'])
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/eiic_rnn_het.p", "wb"))
