from rectipy import Network, random_connectivity
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import poisson
from scipy.signal import find_peaks
import sys
from custom_functions import *

# define parameters
###################

# get sweep condition
rep = int(sys.argv[-1])
g = float(sys.argv[-2])
Delta = float(sys.argv[-3])
path = str(sys.argv[-4])

# model parameters
N = 1000
p = 0.2
C = 50.0
k = 1.0
v_r = -80.0
v_t = -30.0
eta = 0.0
a = 0.01
b = -20.0
d = 150.0
E_e = 0.0
E_i = -65.0
tau_s = 8.0
s_ext = 45.0*1e-3
v_spike = 40.0
v_reset = -55.0
theta_dist = "gaussian"

# define distribution of etas
f = gaussian if theta_dist == "gaussian" else lorentzian
thetas = f(N, mu=v_t, delta=Delta, lb=v_r, ub=2*v_t-v_r)

# define connectivity
W = random_connectivity(N, N, p, normalize=True)

# define inputs
g_in = 10
T = 2500.0
cutoff = 1000.0
start = 1000.0
stop = 1010.0
amp = 20.0*1e-3
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt), N))
inp += poisson.rvs(mu=s_ext*g_in*dt, size=inp.shape)
inp[int((cutoff + start)/dt):int((cutoff + stop)/dt), :] += poisson.rvs(mu=amp*g_in*dt, size=(int((stop-start)/dt), N))
inp = convolve_exp(inp, tau_s, dt)

# run the model
###############

# initialize model
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
             "g_e": 0.0, "g_i": g, "E_e": E_e, "E_i": E_i, "tau_s": tau_s, "v": v_t}

# initialize model
net = Network(dt, device="cpu")
net.add_diffeq_node("ik", f"config/ik_snn/ik", weights=W, source_var="s", target_var="s_i",
                    input_var="g_e_in", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=node_vars.copy(), op="ik_op", spike_reset=v_reset, spike_threshold=v_spike,
                    clear=True)

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False,
              cutoff=int(cutoff/dt))
s = obs.to_dataframe("out")

# calculate dimensionality in the steady-state period
idx_stop = int(start/dts)
s_vals = s.values[:idx_stop, :]
dim_ss = get_dim(s_vals)

# extract spikes in network
spike_counts = []
for idx in range(s.shape[1]):
    peaks, _ = find_peaks(s_vals[:, idx])
    spike_counts.append(peaks)

# calculate firing rate statistics
taus = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
s_mean = np.mean(s.values, axis=1) / tau_s
s_std = np.std(s.values, axis=1) / tau_s
ffs, ffs2 = [], []
for tau in taus:
    ffs.append(fano_factor(spike_counts, s_vals.shape[0], int(tau/dts)))
    ffs2.append(fano_factor2(spike_counts, s_vals.shape[0], int(tau/dts)))

# fit bi-exponential to envelope of impulse response
ir_window = int(300.0/dts)
tau = 10.0
a = 10.0
d = 3.0
p0 = [d, a, tau]
ir = s_mean[idx_stop:] * 1e3
ir = ir - np.mean(ir[ir_window:])
time = s.index.values[int(start/dts):]
time = time - np.min(time)
bounds = ([0.0, 1.0, 1e-1], [1e2, 3e2, 5e2])
p, ir_fit = impulse_response_fit(ir, time, f=alpha, p0=p0, bounds=bounds, gtol=None, loss="linear")

# calculate dimensionality in the impulse response period
ir_window = int(100.0*p[2])
s_vals = s.values[idx_stop:idx_stop+ir_window, :]
dim_ir = get_dim(s_vals)

# # save results
# pickle.dump({"g": g, "Delta": Delta, "theta_dist": theta_dist, "dim_ss": dim_ss, "dim_ir": dim_ir,
#              "s_mean": s_mean, "s_std": s_std, "ff_between": ffs, "ff_within": ffs2, "ff_windows": taus,
#              "ir_target": ir, "ir_fit": ir_fit, "ir_params": p},
#             open(f"{path}/spn_g{int(g)}_D{int(Delta)}_{rep+1}.pkl", "wb"))

# figure settings
print(f"Plotting backend: {plt.rcParams['backend']}")
# plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=False)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0

fig, axes = plt.subplots(nrows=3, figsize=(12, 7.5))

# plotting average firing rate dynamics
ax = axes[1]
ax.plot(s_mean*1e3, label="mean(r)")
ax.plot(s_std*1e3, label="std(r)")
ax.axvline(x=int(start/dts), linestyle="dashed", color="black")
ax.axvline(x=int(stop/dts), linestyle="dashed", color="black")
ax.legend()
ax.set_xlabel("steps")
ax.set_ylabel("r (Hz)")
ax.set_title(f"dim(ss) = {np.round(dim_ss, decimals=1)}")

# plotting impulse response
ax = axes[2]
ax.plot(ir, label="Target IR")
ax.plot(ir_fit, label="Fitted IR")
ax.legend()
ax.set_xlabel("steps")
ax.set_ylabel("r (Hz)")
ax.set_title(f"dim(ir) = {np.round(dim_ir, decimals=1)}, tau(ir) = {np.round(p[2], decimals=1)} ms")

# plotting spiking dynamics
ax = axes[0]
sample_neurons = np.random.choice(np.arange(N), size=(100,), replace=False)
im = ax.imshow(s.iloc[:, sample_neurons].T, aspect="auto", interpolation="none", cmap="Greys")
plt.colorbar(im, ax=ax)
ax.set_xlabel("steps")
ax.set_ylabel("neurons")
ax.set_title("Spiking dynamics")

fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/spn_dynamics.pdf')

plt.show()
