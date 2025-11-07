from rectipy import Network, random_connectivity
import sys
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
import numpy as np
from reservoir_computing.utility_funcs import lorentzian
import matplotlib.pyplot as plt
import pickle


# define parameters
###################

# model parameters
N = 1000
p = 0.2
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 0.1
eta = 0.0
a = 0.03
b = -2.0
d = 10.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# define lorentzian of etas
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2*v_t - v_r)

# define connectivity
# W = random_connectivity(N, N, p, normalize=True)

# define inputs
T = 1500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp = np.zeros((int(T/dt), 1)) + 32.0
inp[int(900/dt):int(1100/dt), 0] += 30.0

# run the model
###############

# initialize model
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_r}

# initialize model
net = Network(dt=dt, device="cpu")
net.add_diffeq_node("rs", f"config/ik_snn/rs", #source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", reset_var="v", spike_var="spike", to_file=False,
                    node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True, N=N)

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), verbose=True, cutoff=int(cutoff/dt))
res = obs.to_dataframe("out")

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
ax.plot(np.mean(res, axis=1))
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel('time')
ax.set_ylim([0.0, 0.45])
ax = axes[1]
ax.imshow(res.T, aspect="auto", interpolation="none", cmap="Greys")
ax.set_ylabel(r'neurons')
ax.set_xlabel('time (ms)')
plt.tight_layout()
fig.canvas.draw()
plt.savefig("/home/richard-gast/Documents/rs_dynamics_1.svg")
plt.show()

# save results
# pickle.dump({'results': res}, open("results/rs_snn_hom_low_sfa.p", "wb"))
