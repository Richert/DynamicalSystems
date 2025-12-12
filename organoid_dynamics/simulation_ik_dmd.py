from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'TkAgg'
from pydmd.utils import pseudo_hankel_matrix
from pydmd.preprocessing import hankel_preprocessing
from pydmd import MrDMD, DMD

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
    for sample in range(num_samples):# Generate white noise (Gaussian)
        x += scale*white_noise[sample] - x / tau
        colored_noise[sample] = x
    return colored_noise

def make_plot(X, x=None, y=None, figsize=(12, 8), title=""):
    """
    Plot of the data X
    """
    plt.figure(figsize=figsize)
    plt.title(title)
    X = np.real(X)
    CS = plt.pcolor(x, y, X)
    cbar = plt.colorbar(CS)
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.show()

# define parameters
###################

# model parameters
C = 100.0
k = 0.7
v_r = -70.0
v_t = -45.0
Delta = 1.0
eta = 85.0
b = -2.0
kappa = 5.0
U0 = 0.6
alpha = 0.0
psi = 300.0
theta = 0.02
g_a = 30.0
g_n = 0.1
g_g = 15.0
tau_w = 50.0
tau_ca = 250.0
tau_u = 100.0
tau_x = 700.0
tau_a = 5.0
tau_n = 150.0
tau_g = 10.0
tau_s = 1.0
node_vars = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'alpha': alpha,
    'g_a': g_a, 'g_n': g_n, 'g_g': g_g, 'b': b, 'U0': U0, 'tau_ca': tau_ca, 'tau_w': tau_w, 'tau_u': tau_u,
    'tau_x': tau_x, 'tau_a': tau_a, 'tau_n': tau_n, 'tau_g': tau_g, 'tau_s': tau_s, 'psi': psi, 'theta': theta
}

# define inputs
T = 5000.0
cutoff = 0.0
dt = 1e-3
dts = 1.0
s_in = 0.0
noise_lvl = 0.5
noise_tau = 50.0/dt
inp = np.zeros((int(T/dt),))
inp[int(1000/dt):int(2000/dt)] += s_in
inp += generate_colored_noise(int(T/dt), tau=noise_tau, scale=noise_lvl*np.sqrt(dt))

# run the mean-field model
##########################

# initialize model
op = "ik_full_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/pc")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in node_vars.items()})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='scipy',
                outputs={'r': f'p/{op}/r', 'w': f'p/{op}/w', 'x': f'p/{op}/x', 'ca': f'p/{op}/ca'},
                inputs={f'p/{op}/I_ext': inp}
                )

# run the DMD analysis
######################

# run DMD fit
delays = 100
max_lvl = 3
X = np.array([res_mf["r"].values])
X[0, :] -= np.mean(X[0, :])
X[0, :] /= np.std(X[0, :])
X= pseudo_hankel_matrix(X, delays)
sub_dmd = DMD(svd_rank=0)
dmd = MrDMD(sub_dmd, max_level=max_lvl, max_cycles=1)
dmd.fit(X=X)

# extract modes
dmd_res = {"mode_real": [], "mode_imag": [], "dynamics_real": [], "dynamics_imag": []}
for lvl in range(max_lvl):
    mode = dmd.partial_modes(level=lvl)
    dyn = dmd.partial_dynamics(level=lvl)
    dmd_res["mode_real"].append(mode.real)
    dmd_res["mode_imag"].append(mode.imag)
    dmd_res["dynamics_real"].append(dyn.real.T)
    dmd_res["dynamics_imag"].append(dyn.imag.T)

# plot results
##############

# simulation
fig, axes = plt.subplots(nrows=5, figsize=(12, 10))
fig.suptitle("Mean-field dynamics")
ax = axes[0]
ax.plot(res_mf.index, res_mf["r"] * 1e3, label="mean-field")
ax.set_ylabel(r'$r(t)$ (Hz)')
ax.set_title("IK (full model) Mean-Field Dynamics")
ax = axes[1]
ax.plot(res_mf.index, res_mf["w"], label="mean-field")
ax.set_ylabel(r'$w(t)$')
ax = axes[2]
ax.plot(res_mf.index, res_mf["x"], label="mean-field")
ax.set_ylabel(r'$x(t)$')
ax = axes[3]
ax.plot(res_mf.index, res_mf["ca"], label="mean-field")
ax.set_ylabel(r'$ca(t)$')
ax = axes[4]
ax.plot(res_mf.index, inp[::int(dts/dt)])
ax.set_ylabel(r'$I(t)$')
ax.set_xlabel("time (ms)")
plt.tight_layout()

# DMD analysis
fig, axes = plt.subplots(nrows=max_lvl, ncols=4, figsize=(12, 10))
for lvl in range(max_lvl):
    for col, key in enumerate(dmd_res.keys()):
        ax = axes[lvl, col]
        ax.plot(dmd_res[key][lvl])
        ax.set_title(f"{key}, lvl = {lvl}")
plt.tight_layout()

plt.show()
# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
