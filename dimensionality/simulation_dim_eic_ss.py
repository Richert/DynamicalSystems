from rectipy import FeedbackNetwork, random_connectivity
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
rep = 0 #int(sys.argv[-1])
g = 30.0 #float(sys.argv[-2])
Delta = 3.0 #float(sys.argv[-3])
lbd = 0.4

# general parameters
v_spike = 1000.0
v_reset = -1000.0
theta_dist = "gaussian"
p = 0.2
N = 1000

# exc parameters
p_e = 0.8
N_e = int(N*p_e)
C_e = 100.0
k_e = 0.7
v_r_e = -60.0
v_t_e = -40.0
eta_e = 50.0
a_e = 0.03
b_e = -2.0
d_e = 40.0
E_e = 0.0
tau_s_e = 6.0

# inh parameters
p_i = 1 - p_e
N_i = int(N*p_i)
C_i = 100.0
k_i = 0.7
v_r_i = -75.0
v_t_i = -55.0
eta_i = 100.0
a_i = 0.03
b_i = -2.0
d_i = 100.0
E_i = -65.0
tau_s_i = 10.0

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
thetas_e = f(N_e, mu=v_t_e, delta=Delta, lb=v_r_e, ub=2*v_t_e-v_r_e)
thetas_i = f(N_i, mu=v_t_i, delta=Delta, lb=v_r_i, ub=2*v_t_i-v_r_i)

# random connectivity
W_ee = random_connectivity(N_e, N_e, p, normalize=True)
W_ie = random_connectivity(N_i, N_e, p, normalize=True)
W_ei = random_connectivity(N_e, N_i, p, normalize=True)
W_ii = random_connectivity(N_i, N_i, p, normalize=True)

# connectivity scaling
g_ee = g*p_e*lbd
g_ei = g*p_e*(1-lbd)
g_ie = g*p_i*(1-lbd)
g_ii = g*p_i*lbd

# define inputs
T = 3000.0
cutoff = 1000.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), 1))

# run the model
###############

# initialize model
exc_vars = {"C": C_e, "k": k_e, "v_r": v_r_e, "v_theta": thetas_e, "eta": eta_e, "tau_u": 1/a_e, "b": b_e, "kappa": d_e,
            "g_e": g_ee, "E_i": E_i, "tau_s": tau_s_e, "v": v_t_e, "g_i": g_ei}
inh_vars = {"C": C_i, "k": k_i, "v_r": v_r_i, "v_theta": thetas_i, "eta": eta_i, "tau_u": 1/a_i, "b": b_i, "kappa": d_i,
            "g_e": g_ie, "E_i": E_i, "tau_s": tau_s_i, "v": v_t_i, "g_i": g_ii}

# initialize model
net = FeedbackNetwork(dt, device="cpu")
net.add_diffeq_node("exc", f"config/ik_snn/ik", weights=W_ee, source_var="s", target_var="s_e",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=exc_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    clear=True)
net.add_diffeq_node("inh", f"config/ik_snn/ik", weights=W_ii, source_var="s", target_var="s_i",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=inh_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    clear=True)
net.add_edge("exc", "inh", weights=W_ie, feedback=False)
net.add_edge("inh", "exc", weights=W_ei, feedback=True)

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False,
              cutoff=int(cutoff/dt), record_vars=[("exc", "s", False)])
s_i = obs.to_numpy("out")
s_e = obs.to_numpy(("exc", "s"))

# calculate dimensionality
cov = s_e.T @ s_e
eigs = np.abs(np.linalg.eigvals(cov))
dim = np.sum(eigs) ** 2 / np.sum(eigs ** 2)

# calculate firing rate statistics
s_mean = np.mean(s_e, axis=1) * 1e3/tau_s_e
s_std = np.std(s_e, axis=1) * 1e3/tau_s_e

# save results
# pickle.dump({"g": g, "Delta": Delta, "theta_dist": theta_dist, "dim": dim, "s_mean": s_mean, "s_std": s_std},
#             open(f"/media/fsmresfiles/richard_data/numerics/dimensionality/bal_ss_g{int(g)}_D{int(Delta*10)}_{rep+1}.p",
#                  "wb"))

# plotting
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(s_mean, label="mean(r_e)")
ax.plot(s_std, label="std(r_e)")
ax.legend()
ax.set_xlabel("steps")
ax.set_ylabel("r_e")
ax.set_title(f"Dim = {dim}")
plt.tight_layout()

_, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(s_e.T, aspect="auto", interpolation="none", cmap="Greys")
plt.colorbar(im, ax=ax)
ax.set_xlabel("steps")
ax.set_ylabel("exc neurons")
plt.tight_layout()

_, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(s_i.T, aspect="auto", interpolation="none", cmap="Greys")
plt.colorbar(im, ax=ax)
ax.set_xlabel("steps")
ax.set_ylabel("inh neurons")
plt.tight_layout()
plt.show()
