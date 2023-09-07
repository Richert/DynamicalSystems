from rectipy import Network, random_connectivity
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import cauchy, norm
from scipy.signal import find_peaks


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def gaussian(n, mu: float, sd: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = norm.rvs(loc=mu, scale=sd)
        while s <= lb or s >= ub:
            s = norm.rvs(loc=mu, scale=sd)
        samples[i] = s
    return samples


# define parameters
###################

# model parameters
N = 2000
p = 0.2
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 0.9
SD = 4.0
eta = 50.0
a = 0.03
b = -2.0
d = 10.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# define lorentzian of etas
thetas_l = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2*v_t-v_r)
thetas_g = gaussian(N, mu=v_t, sd=SD, lb=v_r, ub=2*v_t-v_r)

# define connectivity
W = random_connectivity(N, N, p, normalize=True)

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), 1))

# run the model
###############

thetas_all = {"norm": thetas_g, "lorentz": thetas_l}
results = {"norm": [], "lorentz": []}
for key, thetas in thetas_all.items():

    # initialize model
    node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
                 "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

    # initialize model
    net = Network(dt, device="cuda:0")
    net.add_diffeq_node("ik", f"config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                        input_var="I_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                        node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                        clear=True)

    # perform simulation
    obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False)
    res = obs.to_numpy("out")

    # identify spikes
    spike_rates = []
    for neuron in range(res.shape[1]):
        signal = res[int(cutoff * dts / dt):, neuron]
        signal = signal / np.max(signal)
        peaks, _ = find_peaks(signal, height=0.5, width=5)
        spike_rates.append(len(peaks) * 1e3 / (T - cutoff))

    # store results
    results[key] = (res, spike_rates)

# plot results
fig, axes = plt.subplots(nrows=3, figsize=(10, 6))
ax = axes[0]
ax.hist(results["lorentz"][1], density=False, rwidth=0.75, color="blue", label="Lorentzian")
ax.hist(results["norm"][1], density=False, rwidth=0.75, color="orange", label="Gaussian", alpha=0.8)
ax.set_xlabel("spike rate")
ax.set_ylabel("count")
ax = axes[1]
ax.imshow(results["lorentz"][0].T, interpolation="none", aspect="auto", cmap="Greys")
ax.set_xlabel("time")
ax.set_ylabel("neuron id")
ax.set_title(f"Lorentzian - Delta = {Delta}")
ax = axes[2]
ax.imshow(results["norm"][0].T, interpolation="none", aspect="auto", cmap="Greys")
ax.set_xlabel("time")
ax.set_ylabel("neuron id")
ax.set_title(f"Gaussian - SD = {SD}")
plt.tight_layout()
plt.show()

# save results
# pickle.dump(results, open("results/rs_gauss_lorentz.p", "wb"))
