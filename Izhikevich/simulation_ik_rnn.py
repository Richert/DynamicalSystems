import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu import RNN
from pyrecu.neural_models import ik_ata
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# model parameters
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
cutoff = 100.0
dt = 1e-4
dts = 1e-2
inp = np.zeros((int(T/dt),)) + 60.0
inp[int(600/dt):int(1600/dt)] -= 15.0

# run the model
###############

# initialize model
u_init = np.zeros((3*N,))
u_init[:N] -= 60.0
model = RNN(N, 3*N, ik_ata, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset, d=d, a=a, b=b,
            tau_s=tau_s, J=J, g=g, e_r=e_r, g_e=g_e, u_init=u_init)

# define outputs
outputs = {'s': {'idx': np.arange(2*N, 3*N), 'avg': True}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, parallel=True, decorator=nb.njit,
                verbose=True, fastmath=True)

# plot results
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(np.mean(res["s"], axis=1))
ax.set_xlabel('time')
ax.set_ylabel(r'$s(t)$')
plt.tight_layout()
plt.show()
