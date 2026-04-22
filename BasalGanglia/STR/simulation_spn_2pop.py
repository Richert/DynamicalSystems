import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(2)
import pickle
import numpy as np
from rectipy import Network
import matplotlib.pyplot as plt
from scipy.stats import cauchy, norm, uniform
plt.rcParams['backend'] = 'TkAgg'


def get_param_dist(n: int, eta: float, delta: float, lb: float, ub: float, dist: str = "gaussian"):
    samples = np.zeros((n,))
    if dist == "gaussian":
        f = norm.rvs
    elif dist == "lorentzian":
        f = cauchy.rvs
    else:
        f = uniform.rvs
    for i in range(n):
        s = f(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = f(loc=eta, scale=delta)
        samples[i] = s
    return samples

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

@nb.njit
def integrate_noise(x, inp, scale, tau):
    return x + scale * inp - x / tau


# define parameters
###################

N = 1000
device = "cpu"

# model parameters
C = 50.0
k = 1.0
v_r = -80.0
v_t = -30.0
eta_1 = 100.0
eta_2 = 150.0
Delta_1 = 20.0
Delta_2 = 20.0
kappa = 150.0
a = 0.01
b = -20.0
tau_s = 4.0
g_i = 4.0
E_i = -60.0
v_spike = 500.0
v_reset = -500.0
d1_etas = get_param_dist(int(N*0.5), eta=eta_1, delta=Delta_1, lb=eta_1-10*Delta_1, ub=eta_1+10*Delta_1, dist="gaussian")
d2_etas = get_param_dist(int(N*0.5), eta=eta_2, delta=Delta_2, lb=eta_2-10*Delta_2, ub=eta_2+10*Delta_2, dist="gaussian")

# define inputs
cutoff = 600.0
T = 900.0 + cutoff
dt = 1e-2
dts = 1e-1
noise_scale = 10.0
noise_tau = 100.0
steps = int(T/dt)
inp = np.zeros((steps, 1))
for i in range(inp.shape[1]):
    inp[:, i] = generate_colored_noise(steps, noise_tau, noise_scale)

# collect parameters
d1_vars = {"C": C, "k": k, "v_r": v_r, "tau_u": 1/a, "b": b, "kappa": kappa,
             "g_i": g_i, "E_i": E_i, "tau_s": tau_s, "v": v_t, "eta": d1_etas}

# run the model
###############

# initialize model
net = Network(dt=dt, device=device)
net.add_diffeq_node("SPNs", node=f"config/snn/ik", source_var="s", target_var="s_i",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=d1_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True, N=N, record_vars=["v"])

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, cutoff=int(cutoff/dt),
              record_vars=[("SPNs", "v", True)])
s = obs.to_numpy("out") * 1e3/tau_s
v = obs.to_numpy(("SPNs", "v"))

# save results
# pickle.dump({'s': s, 'v': v}, open("results/spn_snn.pkl", "wb"))

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
im = ax.imshow(s.T, aspect="auto", interpolation="none")
plt.colorbar(im, ax=ax, shrink=0.65)
ax.set_xlabel('time')
ax.set_ylabel('neurons')
ax.set_title("Spiking activity")
ax = axes[1]
time = np.linspace(cutoff, T, int((T-cutoff)/dts))
ax.plot(time, np.mean(s, axis=1) / tau_s)
ax.set_xlabel("time")
ax.set_ylabel("r (Hz)")
ax.set_title("Average synaptic activation")
plt.tight_layout()
plt.show()
