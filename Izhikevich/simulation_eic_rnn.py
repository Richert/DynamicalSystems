import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu import RNN
from pyrecu.neural_models import ik_ei_ata
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


# define parameters
###################

N = 10000

# RS neuron parameters
Ce = 100.0   # unit: pF
ke = 0.7  # unit: None
ve_r = -60.0  # unit: mV
ve_t = -40.0  # unit: mV
ve_spike = 60.0  # unit: mV
ve_reset = -60.0  # unit: mV
Delta_e = 2.0  # unit: mV
de = 20.0
ae = 0.03
be = -2.0

# IB neuron parameters
Ci = 150.0   # unit: pF
ki = 1.2  # unit: None
vi_r = -75.0  # unit: mV
vi_t = -45.0  # unit: mV
vi_spike = 56.0  # unit: mV
vi_reset = -56.0  # unit: mV
Delta_i = 1.0  # unit: mV
di = 30.0
ai = 0.01
bi = 5.0

# synaptic parameters
g_ampa = 1.5
g_gaba = 0.8
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 6.0
tau_gaba = 8.0
k_ee = 10.0
k_ei = 10.0
k_ie = 45.0
k_ii = 15.0

# define lorentzian of etas
spike_thresholds_e = lorentzian(N, eta=ve_t, delta=Delta_e, lb=ve_r, ub=0.0)
spike_thresholds_i = lorentzian(N, eta=vi_t, delta=Delta_i, lb=vi_r, ub=0.0)

# define inputs
T = 4000.0
cutoff = 1000.0
dt = 1e-3
dts = 1e-1
I_ext = np.zeros((int(T/dt), 2))
I_ext[:, 0] = 30.0
I_ext[:, 1] = 200.0
I_ext[int(2000/dt):int(3000/dt), 0] += 20.0

# run the model
###############

# initialize model
u_init = np.zeros((4*N+4,))
u_init[:N] -= 60.0
u_init[2*N:3*N] -= 60.0
model = RNN(N, 6*N, ik_ei_ata, Ce=Ce, Ci=Ci, ke=ke, ki=ki, ve_r=ve_r, vi_r=vi_r, ve_t=spike_thresholds_e,
            vi_t=spike_thresholds_i, ve_spike=ve_spike, vi_spike=vi_spike, ve_reset=ve_reset, vi_reset=vi_reset,
            de=de, di=di, ae=ae, ai=ai, be=be, bi=bi, g_ampa=g_ampa, g_gaba=g_gaba, E_ampa=E_ampa, E_gaba=E_gaba,
            tau_ampa=tau_ampa, tau_gaba=tau_gaba, k_ee=k_ee, k_ei=k_ei, k_ie=k_ie, k_ii=k_ii, u_init=u_init)

# define outputs
outputs = {'se': {'idx': np.asarray([4*N]), 'avg': False}, 'si': {'idx': np.asarray([4*N+1]), 'avg': False}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=I_ext, cutoff=cutoff, parallel=True, fastmath=True)

# plot results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(res["se"])
ax.plot(res["si"])
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel('time')
plt.legend(['RS', 'IB'])
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/eic_rs_rnn_het.p", "wb"))
