import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import pickle
from scipy.stats import cauchy, norm
from rectipy import Network, random_connectivity


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


##############
# parameters #
##############

# choose neuron type
neuron_type = "lts"
distribution_type = "gauss"

if neuron_type == "rs":

    C = 100.0
    k = 0.7
    v_r = -60.0
    v_t = -40.0
    eta = 0.0
    a = 0.03
    b = -2.0
    d = 10.0
    g = 15.0
    E_r = 0.0
    tau_s = 6.0

elif neuron_type == "fs":

    C = 20.0
    k = 1.0
    v_r = -55.0
    v_t = -40.0
    eta = 25.0
    a = 0.2
    b = 0.025
    d = 0.0
    g = 5.0
    E_r = -65.0
    tau_s = 8.0

elif neuron_type == "lts":

    C = 10.0
    k = 1.0
    v_r = -56.0
    v_t = -42.0
    eta = 100.0
    a = 0.03
    b = 8.0
    d = 20.0
    g = 5.0
    E_r = -65.0
    tau_s = 8.0

else:

    raise ValueError("Wrong neuron type")

# network parameters
I_ext = np.linspace(50.0, 100.0, num=100)
v_reset = -1000.0
v_spike = 1000.0
N = 1000
p = 0.2
idx = 1
SD = 0.5
Delta = 0.1
print(f"Condition: {neuron_type}, {distribution_type}, Delta = {Delta}, sigma = {SD}")

# define inputs
ts = 10.0
T = 2100.0*ts
cutoff = 100.0*ts
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), 1))
inp[int(100*ts/dt):int(2100*ts/dt), 0] += np.linspace(0.0, 100.0, num=int(2000*ts/dt))
# inp[int(1100*ts/dt):int(2100*ts/dt), 0] += np.linspace(100.0, 0.0, num=int(1000*ts/dt))

# get connectivity
W = random_connectivity(N, N, p, normalize=True)

# get thetas
if distribution_type == "lorentz":
    thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2*v_t-v_r)
else:
    thetas = gaussian(N, mu=v_t, sd=SD, lb=v_r, ub=2*v_t-v_r)

##############
# simulation #
##############

try:
    results = pickle.load(open(f"results/bifurcations_{neuron_type}_{idx}.pkl", "rb"))
except FileNotFoundError:
    results = {"lorentz": [], "gauss": [], "Delta": Delta, "SD": SD}

# collect parameters
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1 / a, "b": b, "kappa": d,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

# initialize network
net = Network(dt=dt, device="cuda:0")
net.add_diffeq_node("ik", "config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                    node_vars=node_vars, op="rs_op", spike_reset=v_reset, spike_threshold=v_spike)

# run simulation
obs = net.run(inp, sampling_steps=int(dts / dt), record_output=True, verbose=False, enable_grad=False)

# store results
results[distribution_type] = np.mean(obs.to_numpy("out"), axis=1)

# save results
pickle.dump(results, open(f"results/bifurcations_{neuron_type}_{idx}.p", "wb"))

############
# plotting #
############

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(results[distribution_type], label=rf"$\Delta_v = {Delta}$ mV, $\sigma_v = {SD}$")
ax.set_xlabel("time")
ax.set_ylabel("s")
ax.legend()
ax.set_title(distribution_type)
plt.tight_layout()
plt.show()