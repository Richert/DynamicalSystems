from rectipy import Network, random_connectivity
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import find_peaks


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
N = 1000
p = 0.2
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
deltas = np.linspace(0.1, 3.1, num=10)
eta = 50.0
a = 0.03
b = -2.0
d = 80.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# define connectivity
W = random_connectivity(N, N, p, normalize=True)

# define inputs
T = 5500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), 1))

# run the model
###############

results = {"Delta": [], "s": [], "rates": []}
for Delta in deltas:

    # initialize spike thresholds
    thetas = gaussian(N, mu=v_t, sd=Delta, lb=v_r, ub=2 * v_t - v_r)

    # initialize model
    node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
                 "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

    # initialize model
    net = Network(dt, device="cpu")
    net.add_diffeq_node("ik", f"config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                        input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                        node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                        clear=True)

    # perform simulation
    obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False)
    res = obs.to_numpy("out")[int(cutoff * dts / dt):]

    # identify spikes
    spike_rates = []
    for neuron in range(res.shape[1]):
        signal = res[:, neuron]
        signal = signal / np.max(signal)
        peaks, _ = find_peaks(signal, height=0.5, width=5)
        spike_rates.append(len(peaks) * 1e3 / (T - cutoff))

    # store results
    results["Delta"].append(Delta)
    results["s"].append(res[int(3000.0*dts/dt):])
    results["rates"].append(np.mean(spike_rates))

# plot results
fig, axes = plt.subplots(nrows=4, figsize=(12, 9))
ax = axes[0]
ax.plot(results["Delta"], results["rates"], color="blue")
ax.set_xlabel("Delta")
ax.set_ylabel("fr")
for i, idx in enumerate([0, 5, 9]):
    ax = axes[i+1]
    ax.imshow(results["s"][idx].T, interpolation="none", aspect="auto", cmap="Greys")
    ax.set_xlabel("time")
    ax.set_ylabel("neuron id")
    ax.set_title(f"Delta = {results['Delta'][idx]}")
plt.tight_layout()
plt.show()

# save results
# pickle.dump(results, open("results/rs_gauss_lorentz.p", "wb"))
