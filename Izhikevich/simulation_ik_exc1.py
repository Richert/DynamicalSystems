import numba as nb
nb.config.THREADING_LAYER = 'omp'
import numpy as np
from pyrecu import ik2_ata, RNN
import matplotlib.pyplot as plt
from time import perf_counter
import pickle
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

N = 10000
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
v_spike = 400.0  # unit: mV
v_reset = -600.0  # unit: mV
v_delta = 1.6  # unit: mV
d = 100.0
a = 0.03
b = -2.0
tau_s = 6.0
J = 1.0
g = 20.0
g_e = 0.0
e_r = 0.0

# define lorentzian of etas
spike_thresholds = v_t+v_delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# define inputs
T = 2100.0
dt = 1e-4
dts = 1e-2
inp = np.zeros((int(T/dt),)) + 75.0
inp[int(600/dt):int(1600/dt)] -= 30.0

# perform simulation
u_init = np.zeros((4*N,))
u_init[0:N] -= 60.0
model = RNN(N, 3*N, ik2_ata, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset, d=d, a=a, b=b,
            tau_s=tau_s, J=J, g=g, e_r=e_r, g_e=g_e, u_init=u_init)
t0 = perf_counter()
res = model.run(T=T, dt=dt, dts=dts, outputs=(np.arange(3*N, 4*N), np.arange(0, N)), inp=inp, cutoff=100.0)
t1 = perf_counter()
print(f"Simulation finished after {t1-t0} s.")
r, v = res[0], res[1]

# plot results
fig, axes = plt.subplots(nrows=3, figsize=(12, 8))

axes[0].plot(np.mean(r, axis=1))
axes[0].set_ylabel('r')
axes[1].plot(np.mean(v, axis=1))
axes[1].set_ylabel('v')

spiking_data = []
for i in np.random.choice(np.arange(r.shape[-1]), size=100, replace=False):
    spikes = np.argwhere(r[:, i] > 0).squeeze()
    spiking_data.append(spikes if spikes.shape else np.asarray([]))

axes[2].eventplot(spiking_data, colors='k', lineoffsets=1.0, linelengths=1.0, linewidths=0.3)
plt.tight_layout()
plt.show()

