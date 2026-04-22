import matplotlib.pyplot as plt
import numpy as np
from pyrates import CircuitTemplate, clear
from numba import njit

def generate_colored_noise(num_samples, tau, scale=1.0):
    """
    Generates Brownian noise by integrating white noise.

    Args:
        num_samples (int): The number of samples in the output Brownian noise.
        scale (float): A scaling factor for the noise amplitude.

    Returns:
        numpy.ndarray: An array containing the generated Brownian noise.
    """
    white_noise = np.random.randn(num_samples)
    x = 0.0
    colored_noise = np.zeros_like(white_noise)
    for sample in range(num_samples):
        x = integrate_noise(x, white_noise[sample], scale, tau)
        colored_noise[sample] = x
    return colored_noise

@njit
def integrate_noise(x, inp, scale, tau):
    return x + scale * inp - x / tau

# load model template
template = CircuitTemplate.from_yaml(f'config/mf/str')

# update template parameters
node_vars = {
    "d1_spn/spn_op/eta": 30.0, "d2_spn/spn_op/eta": 30.0, "fsi/fsi_op/eta": 100.0,
    "d1_spn/spn_op/Delta": 5.0, "d2_spn/spn_op/Delta": 6.0, "fsi/fsi_op/Delta": 3.0,
    "d1_spn/spn_op/g_i": 10.0, "d2_spn/spn_op/g_i": 10.0, "fsi/fsi_op/g_i": 10.0,
    "d1_spn/spn_op/E_i": -60.0, "d2_spn/spn_op/E_i": -60.0, "fsi/fsi_op/E_i": -60.0,
    #"d1_spn/spn_op/tau_r": 2.0, "d2_spn/spn_op/tau_r": 2.0, "fsi/fsi_op/tau_r": 2.0,
    "d1_spn/spn_op/tau_s": 8.0, "d2_spn/spn_op/tau_s": 8.0, "fsi/fsi_op/tau_s": 8.0,
}
template.update_var(node_vars=node_vars)

# set input parameters
cutoff = 100.0
T = 1000.0 + cutoff
dt = 1e-3
dts = 1e-1
noise_tau = 200.0
noise_scale = 50.0

# generate inputs
steps = int(T/dt)
inp_d1 = generate_colored_noise(steps, noise_tau, noise_scale)
inp_d2 = generate_colored_noise(steps, noise_tau, noise_scale)
inp_fsi = generate_colored_noise(steps, noise_tau, noise_scale)
backend = 'default'
solver = 'euler'
out_vars = ['d1_spn', 'd2_spn', 'fsi']
kwargs = {'vectorize': False, 'float_precision': 'float64', 'decorator': njit, 'fastmath': False}

# perform simulation
results = template.run(simulation_time=T, step_size=dt, backend=backend, solver=solver, sampling_step_size=dts,
                       inputs={'d1_spn/spn_op/I_ext': inp_d1, 'fsi/fsi_op/I_ext': inp_fsi,
                               'd2_spn/spn_op/I_ext': inp_d2},
                       outputs={v: f'{v}/{v[-3:]}_op/r' for v in out_vars}, cutoff=cutoff, **kwargs)
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
