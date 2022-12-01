from pyrates import CircuitTemplate, clear
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'


def correct_input(inp: float, v_t: float, v_r: float):
    if inp <= 0.0:
        return inp
    return inp*np.pi**2/(np.arctan(v_t/np.sqrt(inp)) - np.arctan(v_r/np.sqrt(inp)))**2


# define parameters
###################

# model parameters
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
v_spike = 50.0  # unit: mV
v_reset = -70.0  # unit: mV
Delta = 0.5  # unit: mV
d = 10.0
a = 0.03
b = -2.0

# sxynaptic parameters
tau_s = 6.0
J = 0.0
g = 0.0
q = 0.0
E_r = 0.0

# define inputs
T = 2000.0
cutoff = 1000.0
dt = 1e-3
dts = 1e-1
steps = int(T / dt)

# initialize model
ik = CircuitTemplate.from_yaml("config/ik/ik")

# update parameters
ik.update_var(node_vars={'p/ik_op/C': C, 'p/ik_op/k': k, 'p/ik_op/v_r': v_r, 'p/ik_op/v_t': v_t, 'p/ik_op/v_p': v_spike,
                         'p/ik_op/v_z': np.abs(v_reset), 'p/ik_op/Delta': Delta, 'p/ik_op/d': d, 'p/ik_op/a': a,
                         'p/ik_op/b': b, 'p/ik_op/tau_s': tau_s, 'p/ik_op/g': g, 'p/ik_op/q': q, 'p/ik_op/E_r': E_r})

Is = np.asarray([correct_input(inp, v_t=v_spike, v_r=v_reset) for inp in np.arange(0, 200.0, 10)])
results = []
for mu in Is:

    print('')
    print(f'Running simulation for I = {mu}.')

    # define inputs
    inp = np.zeros((steps,)) + mu

    # run simulation
    res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, outputs={'s': 'p/ik_op/s'}, in_place=False,
                 solver='scipy', cutoff=cutoff, inputs={'p/ik_op/I_ext': inp})
    clear(ik)
    results.append(np.mean(res['s'].v1))

# save results
pickle.dump({'results': results, 'inputs': Is}, open("results/rs_fre_io.p", "wb"))

# plot results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(Is, results)
plt.show()
