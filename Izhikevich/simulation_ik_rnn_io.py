import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu import RNN
from pyrecu.neural_models import ik_ata
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

# neuron parameters
C = 150.0   # unit: pF
k = 1.2  # unit: None
v_r = -75.0  # unit: mV
v_t = -45.0  # unit: mV
v_spike = 56.0  # unit: mV
v_reset = -56.0  # unit: mV
Delta = 1.0  # unit: mV
d = 60.0
a = 0.01
b = 5.0

# synaptic parameters
g = 1.5
E = -65.0
tau = 6.0
J = 0.0

# define lorentzian of etas
spike_thresholds = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=0.0)

# define inputs
T = 1000.0
cutoff = 0.0
dt = 1e-3
dts = 1e-1

# run the model
###############

# initialize model
u_init = np.zeros((3*N,))
u_init[:N] += v_r
model = RNN(N, 3*N, ik_ata, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset, d=d, a=a, b=b,
            tau_s=tau, J=J, g=g, E_r=E, q=0.0, u_init=u_init)

# define outputs
outputs = {'s': {'idx': np.arange(2*N, 3*N), 'avg': True}}

# loop over different input strengths
Is = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
results = []
for mu in Is:

    print(f'Running simulation for I = {mu}.')
    print('')

    # define inputs
    inp = np.zeros((int(T / dt),)) + mu

    # run simulation
    res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, parallel=True, fastmath=True)
    results.append(res['s'][-1])

# plot results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(Is, results)
plt.show()
