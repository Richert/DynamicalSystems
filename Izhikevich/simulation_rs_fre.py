from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb
from rectipy import Network, random_connectivity
import sys
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
from reservoir_computing.utility_funcs import lorentzian


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
v_spike = 2000.0
v_reset = -2000.0

# define inputs
T = 1500.0
cutoff = 500.0
dt = 5e-3
dts = 1e-1
inp = np.zeros((int(T/dt), 1)) + 27.0
inp[int(900/dt):int(1100/dt), 0] += 30

# run the mean-field model
##########################

# initialize model
ik = CircuitTemplate.from_yaml("config/ik_mf/ik")

# update parameters
ik.update_var(node_vars={'p/ik_op/C': C, 'p/ik_op/k': k, 'p/ik_op/v_r': v_r, 'p/ik_op/v_t': v_t, 'p/ik_op/Delta': Delta,
                         'p/ik_op/d': d, 'p/ik_op/a': a, 'p/ik_op/b': b, 'p/ik_op/tau_s': tau_s, 'p/ik_op/g': g,
                         'p/ik_op/E_r': E_r, 'p/ik_op/v': v_r})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='heun',
                outputs={'s': 'p/ik_op/s'}, inputs={'p/ik_op/I_ext': inp[:, 0]})

# run the SNN model
###################

# define lorentzian of etas
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=0.0)

# define connectivity
W = random_connectivity(N, N, p, normalize=True)

# initialize model
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_r}

# initialize model
net = Network(dt=dt, device="cpu")
net.add_diffeq_node("snn", f"config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                    verbose=False, clear=True, device="cpu")

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=True, cutoff=int(cutoff/dt))
res_snn = obs.to_dataframe("out")

print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "sans"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (6, 3)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.5
markersize = 2

# plot results
fig, axes = plt.subplots(nrows=2)
ax = axes[0]
ax.plot(res_mf.index, 1e3*res_mf["s"] / tau_s, label="Mean-Field")
ax.plot(res_mf.index, 1e3*np.mean(res_snn, axis=1) / tau_s, label="SNN")
ax.set_ylim([0.0, 75.0])
ax.set_ylabel(r'spike rate (Hz)')
ax.set_xlabel("time (ms)")
ax.legend()
ax = axes[1]
ax.imshow(res_snn.T, aspect="auto", interpolation="none", cmap="Greys")
plt.tight_layout()
fig.canvas.draw()
plt.savefig("/home/richard-gast/Documents/rs_dynamics_6.svg")
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
