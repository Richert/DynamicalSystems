import matplotlib.pyplot as plt
import numpy as np
from pyrates import CircuitTemplate, clear
from numba import njit

# load model template
template = CircuitTemplate.from_yaml(f'config/mf/str')

# update template parameters
node_vars = {
    "d1_spn/spn_op/eta": 300.0, "d2_spn/spn_op/eta": 250.0, "fsi/fsi_op/eta": 100.0,
    "d1_spn/spn_op/Delta": 0.4, "d2_spn/spn_op/Delta": 0.3, "fsi/fsi_op/Delta": 0.8,
    "d1_spn/spn_op/g_i": 10.0, "d2_spn/spn_op/g_i": 10.0, "fsi/fsi_op/g_i": 10.0,
    "d1_spn/spn_op/E_i": -60.0, "d2_spn/spn_op/E_i": -60.0, "fsi/fsi_op/E_i": -60.0,
    "d1_spn/spn_op/tau_r": 2.0, "d2_spn/spn_op/tau_r": 2.0, "fsi/fsi_op/tau_r": 2.0,
    "d1_spn/spn_op/tau_d": 8.0, "d2_spn/spn_op/tau_d": 8.0, "fsi/fsi_op/tau_d": 8.0,
}
template.update_var(node_vars=node_vars)

# set pyrates-specific parameters
cutoff = 100.0
T = 1000.0 + cutoff
start = 200.0 + cutoff
stop = 800.0 + cutoff
dt = 1e-3
dts = 1e-1
inp_d1 = np.zeros((int(T/dt),))
inp_d1[int(start/dt):int(stop/dt)] += 200.0
inp_d2 = np.zeros_like(inp_d1)
inp_d2[int(start/dt):int(stop/dt)] += 200.0
inp_fsi = np.zeros_like(inp_d1)
inp_fsi[int(start/dt):int(stop/dt)] += 50.0
backend = 'default'
solver = 'euler'
out_vars = ['d1_spn', 'd2_spn', 'fsi']
kwargs = {'vectorize': False, 'float_precision': 'float64', 'decorator': njit, 'fastmath': False}

# perform simulation
results = template.run(simulation_time=T, step_size=dt, backend=backend, solver=solver, sampling_step_size=dts,
                       inputs={'d1_spn/spn_op/I_ext': inp_d1, 'fsi/fsi_op/I_ext': inp_fsi,
                               'd2_spn/spn_op/I_ext': inp_d2},
                       outputs={v: f'{v}/{v[-3:]}_op/r' for v in out_vars}, cutoff=cutoff)
clear(template)

# plot simulated signal
fig, axes = plt.subplots(figsize=(10, 2*len(out_vars)), nrows=len(out_vars))
for i, v in enumerate(out_vars):
    ax = axes[i]
    ax.plot(results[v]*1e3)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel("r (Hz)")
    ax.set_title(v)
plt.tight_layout()
plt.show()
