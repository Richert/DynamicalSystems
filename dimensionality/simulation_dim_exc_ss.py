from rectipy import Network, random_connectivity
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import cauchy, norm
import sys


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

# get sweep condition
rep = int(sys.argv[-1])
g = float(sys.argv[-2])
Delta = float(sys.argv[-3])

# model parameters
N = 1000
p = 0.2
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
eta = 50.0
a = 0.03
b = -2.0
d = 40.0
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
T = 3000.0
cutoff = 1000.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), 1))

# run the model
###############

# initialize model
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
             "g_e": g, "E_e": E_r, "tau_s": tau_s, "v": v_t, "g_i": 0.0}

# initialize model
net = Network(dt, device="cpu")
net.add_diffeq_node("ik", f"config/ik_snn/ik", weights=W, source_var="s", target_var="s_e",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=node_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    clear=True)

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False,
              cutoff=int(cutoff/dt))
s = obs.to_numpy("out")

# calculate dimensionality
cov = s.T @ s
eigs = np.abs(np.linalg.eigvals(cov))
dim = np.sum(eigs) ** 2 / np.sum(eigs ** 2)

# calculate firing rate statistics
s_mean = np.mean(s, axis=1) / tau_s
s_std = np.std(s, axis=1) / tau_s

# save results
pickle.dump({"g": g, "Delta": Delta, "theta_dist": theta_dist, "dim": dim, "s_mean": s_mean, "s_std": s_std},
            open(f"/media/fsmresfiles/richard_data/numerics/dimensionality/exc_ss_g{int(g)}_D{int(Delta*10)}_{rep+1}.p",
                 "wb"))

# # plotting
# fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(s_mean*1e3/tau_s, label="mean(r)")
# ax.plot(s_std*1e3/tau_s, label="std(r)")
# ax.legend()
# ax.set_xlabel("steps")
# ax.set_ylabel("r")
# ax.set_title(f"Dim = {dim}")
# plt.tight_layout()
# plt.show()
