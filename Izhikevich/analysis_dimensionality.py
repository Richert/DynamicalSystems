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


def gaussian(n, mu: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = norm.rvs(loc=mu, scale=delta)
        while s <= lb or s >= ub:
            s = norm.rvs(loc=mu, scale=delta)
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
Delta = 1.0
eta = 50.0
a = 0.03
b = -2.0
d = 100.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0
theta_dist = "gaussian"

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
thetas = f(N, mu=v_t, delta=Delta, lb=v_r, ub=2*v_t-v_r)

# define connectivity
W = random_connectivity(N, N, p, normalize=True)

# define inputs
T = 2500.0
cutoff = 500.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), 1))

# define parameter sweep
gs = np.linspace(0.0, 20.0, num=5)

# run the model
###############

results = {"g": gs, "dim": [], "s_mean": [], "s_var": []}
for g in gs:

    print(f"Starting calculations for g = {g}")

    # initialize model
    node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
                 "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

    # initialize model
    net = Network(dt, device="cuda:0")
    net.add_diffeq_node("ik", f"config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                        input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                        node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                        clear=True)

    # perform simulation
    obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False)
    s = obs.to_numpy("out")
    print("Model simulation finished.")

    # calculate dimensionality
    cov = s.T @ s
    eigs = np.abs(np.linalg.eigvals(cov))
    results["dim"].append(np.sum(eigs) ** 2 / np.sum(eigs ** 2))
    print("Dimensionality calculation finished.")

    # calculate network statistics
    results["s_mean"].append(np.mean(s, axis=1))
    results["s_var"].append(np.var(s, axis=1))
    print(f"All calculations for g = {g} finished.")

# plot results
fig, axes = plt.subplots(nrows=3, figsize=(12, 8))
ax = axes[0]
ax.plot(results["gs"], results["dim"])
ax.set_xlabel("g")
ax.set_ylabel("dim")
ax.set_title("Participation Ratio")
ax = axes[1]
for g, s in zip(results["gs"], results["s_mean"]):
    ax.plot(s, label=f"g = {np.round(g, decimals=1)}")
ax.set_xlabel("steps")
ax.set_ylabel("mean(s)")
ax.legend()
ax.set_title("Average of network dynamics")
ax = axes[2]
for g, s in zip(results["gs"], results["s_var"]):
    ax.plot(s, label=f"g = {np.round(g, decimals=1)}")
ax.set_xlabel("steps")
ax.set_ylabel("var(s)")
ax.legend()
ax.set_title("Variance of network dynamics")
plt.tight_layout()
plt.show()

# save results
# pickle.dump(results, open("results/rs_gauss_lorentz.p", "wb"))
