from rectipy import Network, circular_connectivity
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import rv_discrete, cauchy


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def dist(x: int, method: str = "inverse", zero_val: float = 1.0, inverse_pow: float = 1.0) -> float:
    if method == "inverse":
        return 1/x**inverse_pow if x > 0 else zero_val
    if method == "exp":
        return np.exp(-x) if x > 0 else zero_val
    else:
        raise ValueError("Invalid method.")


# define parameters
###################

# model parameters
N = 2000
p = 0.2
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.5
eta = 30.0
a = 0.03
b = -2.0
d = 10.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# define lorentzian of etas
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2 * v_t - v_r)

# define connectivity
indices = np.arange(0, N, dtype=np.int32)
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=1.5) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)
# plt.imshow(W, interpolation="none", aspect="equal")
# plt.show()
# print(np.sum(np.sum(W, axis=1)))

# define inputs
cutoff = 500.0
T = 3000.0 + cutoff
dt = 1e-2
dts = 1e-1
p_in = 0.25
n_inputs = int(N * p_in)
center = int(N*0.5)
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))
inp = np.zeros((int(T/dt), N))
inp[:int(200/dt), :] -= 30.0
inp[int(1000/dt):int(1500/dt), inp_indices] += 30.0

# run the model
###############

# initialize model
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

# initialize model
net = Network(dt=dt, device="cuda:0")
net.add_diffeq_node("rs", f"config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                    node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True)

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False)
res = obs.to_dataframe("out")

# save results
pickle.dump({"results": res.iloc[int(cutoff/dts):, :]}, open("/media/richard/data/snn_bump_het.pkl", "wb"))

# plot results
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(res.T, aspect=4.0, interpolation="none")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xlabel('time')
ax.set_ylabel('neurons')
plt.tight_layout()
plt.show()

