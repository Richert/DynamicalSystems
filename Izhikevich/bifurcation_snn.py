import numpy as np
import pickle
from scipy.stats import cauchy, norm
from rectipy import Network, random_connectivity
import matplotlib.pyplot as plt


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

# load data
mapping = pickle.load(open("results/norm_lorentz_fit.pkl", "rb"))

# extract arguments passed to the script
neuron_type = "rs"
idx = 10
n = 50

# choose neuron type
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
    Deltas = np.linspace(0.01, 4.0, num=n)

elif neuron_type == "rs2":

    C = 100.0
    k = 0.7
    v_r = -60.0
    v_t = -40.0
    eta = 0.0
    a = 0.03
    b = -2.0
    d = 100.0
    g = 15.0
    E_r = 0.0
    tau_s = 6.0
    Deltas = np.linspace(0.01, 1.8, num=n)

elif neuron_type == "fs":

    C = 20.0
    k = 1.0
    v_r = -55.0
    v_t = -40.0
    eta = 20.0
    a = 0.2
    b = 0.025
    d = 0.0
    g = 5.0
    E_r = -65.0
    tau_s = 8.0
    Deltas = np.linspace(0.01, 0.8, num=n)

elif neuron_type == "lts":

    C = 100.0
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
    Deltas = np.linspace(0.01, 0.6, num=n)

else:

    raise ValueError("Wrong neuron type")

# get SDs corresponding to Deltas
sd_min = Deltas[0]
sd_max = mapping["norm"][np.argmin(np.abs(mapping["lorentz"] - Deltas[-1]))]
SDs = np.linspace(sd_min, sd_max, num=n)

# network parameters
I_ext = np.linspace(50.0, 100.0, num=100)
v_reset = -1000.0
v_spike = 1000.0
N = 1000
p = 0.2
Delta = Deltas[idx]
SD = SDs[idx]

# define inputs
ts = 10.0
T = 2100.0*ts
dt = 1e-2
dts = 1e-1
cutoff = 100.0*ts
inp = np.zeros((int(T/dt), 1))
if neuron_type == "rs":
    inp[int(100*ts/dt):int(1100*ts/dt), 0] += np.linspace(0.0, 80.0, num=int(1000*ts/dt))
    inp[int(1100*ts/dt):int(2100*ts/dt), 0] += np.linspace(80.0, 0.0, num=int(1000*ts/dt))
elif neuron_type == "rs2":
    inp[int(100 * ts / dt):int(2100 * ts / dt), 0] += np.linspace(0.0, 80.0, num=int(2000 * ts / dt))
else:
    inp[int(100 * ts / dt):int(2100 * ts / dt), 0] += np.linspace(0.0, 120.0, num=int(2000 * ts / dt))

# get connectivity
W = random_connectivity(N, N, p, normalize=True)

# get thetas
thetas_l = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2*v_t-v_r)
thetas_g = gaussian(N, mu=v_t, sd=SD, lb=v_r, ub=2*v_t-v_r)

##############
# simulation #
##############

results = {"lorentz": [], "gauss": [], "Delta": Delta, "SD": SD, "I_ext": inp[int(cutoff/dt)::int(dts/dt), 0] + eta}

for distribution_type, thetas in zip(["lorentz", "gauss"], [thetas_l, thetas_g]):

    # collect parameters
    node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1 / a, "b": b, "kappa": d,
                 "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

    # initialize network
    net = Network(dt=dt, device="cpu")
    net.add_diffeq_node("ik", "config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                        input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                        node_vars=node_vars, op="rs_op", spike_reset=v_reset, spike_threshold=v_spike)

    # run simulation
    obs = net.run(inp, sampling_steps=int(dts / dt), record_output=True, verbose=False, enable_grad=False)

    # store results
    results[distribution_type] = obs.to_numpy("out")[int(cutoff/dts)::, :]

# save results
pickle.dump(results, open(f"results/spikes_{neuron_type}.pkl", "wb"))

############
# plotting #
############

fig, axes = plt.subplots(nrows=2, figsize=(12, 5))
for ax, dist in zip(axes, ["lorentz", "gauss"]):
    ax.imshow(results[dist].T, aspect="auto", cmap="Greys", interpolation="none")
    ax.set_title(rf"$\Delta_v = {Delta}$ mV, $\sigma_v = {SD}$")
ax.set_xlabel("time")
plt.tight_layout()
plt.show()
