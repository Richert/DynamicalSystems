import matplotlib.pyplot as plt
import numpy as np
from pyrates import CircuitTemplate
from scipy.ndimage import gaussian_filter1d

# model parameters
c = 0.1
H_e = 3.25
H_i = 2.0
tau_e = 10.0
tau_i = 200.0
m_max = 1.0
r = 560.0
v_thr = 6.0
ei_ratio = 4.0
noise_lvl = 0.1
noise_sigma = 100.0
u = 0.22
params = {
    'c': c, 'h_e': H_e, 'h_i': H_i, 'tau_e': tau_e, 'tau_i': tau_i, 'm_max': m_max, 'r': r, 'v_thr': v_thr,
    'ei_ratio': ei_ratio
}

# initialize model template and set fixed parameters
op = "jrc_op"
template = CircuitTemplate.from_yaml(f"model_templates.neural_mass_models.jansenrit.JRC2")
template.update_var(node_vars={f"jrc/{op}/{key}": val for key, val in params.items()})

# prepare simulation
T = 9000.0
dt = 1e-1
dts = 1.0
inp = np.zeros((int(T/dt),)) + u
noise = noise_lvl * np.random.randn(inp.shape[0])
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp += noise

# run simulation
res = template.run(simulation_time=T, sampling_step_size=dts, step_size=dt, solver="heun", backend="numpy",
                   inputs={f"jrc/{op}/u": inp}, outputs={"V_e": f"jrc/{op}/V_e", "V_i": f"jrc/{op}/V_i"})

PC = res["V_e"] - res["V_i"]
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(PC)
ax.set_xlabel("time (ms)")
ax.set_ylabel("V (mV)")
plt.show()
