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
E_r = float(sys.argv[-4])

# model parameters
N = 1000
p = 0.2
C = 100.0
k = 0.7
v_r = -70.0
v_t = -50.0
eta = 100.0
a = 0.03
b = -2.0
d = 50.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0
theta_dist = "gaussian"

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
margin = v_t - v_r
v_ts = f(N, mu=v_t, delta=Delta, lb=v_t - margin, ub=v_t + margin)
v_rs = f(N, mu=v_r, delta=Delta, lb=v_r - margin, ub=v_r + margin)
for idx in range(N):
    if v_rs[idx] > v_ts[idx]:
        v_r_old = float(v_rs[idx])
        v_rs[idx] = v_ts[idx]
        v_ts[idx] = v_r_old

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
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": v_ts, "eta": eta, "tau_u": 1 / a, "b": b, "kappa": d,
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
path = "/media/fsmresfiles/richard_data/numerics/dimensionality"
pickle.dump({"g": g, "Delta": Delta, "theta_dist": theta_dist, "dim": dim, "s_mean": s_mean, "s_std": s_std,
             "E_r": E_r}, open(f"{path}/rev_ss_g{int(g)}_D{int(Delta*10)}_E{int(abs(E_r))}_{rep+1}.p", "wb"))

# # plotting average population dynamics
# _, ax = plt.subplots(figsize=(12, 4))
# ax.plot(s_mean*1e3, label="mean(r)")
# ax.plot(s_std*1e3, label="std(r)")
# ax.legend()
# ax.set_xlabel("steps")
# ax.set_ylabel("r")
# ax.set_title(f"Dim = {dim}")
# plt.tight_layout()
#
# # plotting spikes
# _, ax = plt.subplots(figsize=(12, 4))
# im = ax.imshow(s.T, aspect="auto", interpolation="none", cmap="Greys")
# plt.colorbar(im, ax=ax)
# ax.set_xlabel("steps")
# ax.set_ylabel("neurons")
# plt.tight_layout()
#
# # plotting spike threshold distribution and fraction of neurons that receive exc vs. inh input
# margin = 25.0
# vs = np.linspace(v_r - margin, v_t + margin, num=N)
# p_thresh = norm.pdf(vs, loc=v_t, scale=Delta)
# p_rest = norm.pdf(vs, loc=v_r, scale=Delta)
# inh_frac = np.mean(E_r < v_rs)
# exc_frac = np.mean(E_r > v_ts)
# bal_frac = 1.0 - inh_frac - exc_frac
# fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
# ax = axes[0]
# ax.plot(vs, p_thresh, label="spike thresholds")
# ax.plot(vs, p_rest, label="resting potentials")
# ax.scatter([E_r], [0.0], s=60.0, c="black", label="reversal potential", marker="d")
# ax.axvline(x=E_r, ymin=0, ymax=1, color="black", linestyle="dashed")
# ax.set_xlabel("v (mV)")
# ax.set_ylabel("p")
# ax.set_title("Distribution of spike thresholds")
# ax = axes[1]
# ax.bar([1, 2, 3], [inh_frac, bal_frac, exc_frac], width=0.75, align="center")
# ax.set_xticks([1, 2, 3], labels=["inh.", "bal.", "exc."])
# ax.set_ylabel("p")
# ax.set_title("Distribution of synapse types")
# plt.tight_layout()
# plt.show()
