import numpy as np
from rectipy import Network
from scipy.stats import cauchy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
plt.rcParams['backend'] = 'TkAgg'


def FWHM(s: np.ndarray, plot: bool, n_bins: int = 500) -> float:
    center, width = cauchy.fit(s, loc=np.mean(s), scale=np.var(s))
    if plot:
        bins = np.linspace(np.min(s), np.max(s), n_bins)
        y, _ = np.histogram(s, bins)
        y = np.asarray(y, dtype=np.float64) / np.sum(y)
        x = np.asarray([(bins[i + 1] + bins[i]) / 2.0 for i in range(n_bins - 1)])
        ymax = 1.2 * np.max(y)
        xrange = np.max(bins) - np.min(bins)
        fig, ax = plt.subplots()
        ax.plot(x, y, label="data")
        ax.plot(x, cauchy.pdf(x, loc=center, scale=width), label="fit")
        ax.legend()
        ax.axvline(x=center - width, ymin=0.0, ymax=1.0, color="red")
        ax.axvline(x=center + width, ymin=0.0, ymax=1.0, color="red")
        ax.set_xlim([center - xrange / 6, center + xrange / 6])
        ax.set_ylim([0.0, ymax])
        plt.show()
    return width


def get_fwhm(signal: np.ndarray, n_bins: int = 500, plot_steps: int = 1000, jobs: int = 8) -> np.ndarray:
    pool = Parallel(n_jobs=jobs)
    widths = pool(delayed(FWHM)(signal[i, :], i+1 % plot_steps == 0, n_bins) for i in range(signal.shape[0]))
    return np.asarray(widths)


# define parameters
###################

# condition
cond = "strong_sfa_2"
cond_map = {
    "no_sfa_1": {"kappa": 0.0, "eta": 0.0, "eta_inc": 30.0, "eta_init": -30.0, "b": -5.0, "delta": 5.0},
    "weak_sfa_1": {"kappa": 100.0, "eta": 0.0, "eta_inc": 35.0, "eta_init": 0.0, "b": -5.0, "delta": 5.0},
    "strong_sfa_1": {"kappa": 300.0, "eta": 0.0, "eta_inc": 45.0, "eta_init": 0.0, "b": -5.0, "delta": 5.0},
    "no_sfa_2": {"kappa": 0.0, "eta": -150.0, "eta_inc": 190.0, "eta_init": -50.0, "b": -20.0, "delta": 5.0},
    "weak_sfa_2": {"kappa": 100.0, "eta": -20.0, "eta_inc": 70.0, "eta_init": -100.0, "b": -20.0, "delta": 5.0},
    "strong_sfa_2": {"kappa": 300.0, "eta": 40.0, "eta_inc": 100.0, "eta_init": 0.0, "b": -20.0, "delta": 5.0},
}


# model parameters
N = 2000
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
eta = 0.0  # unit: pA
Delta = cond_map[cond]["delta"]
kappa = cond_map[cond]["kappa"]
tau_u = 35.0
b = cond_map[cond]["b"]
tau_s = 6.0
tau_x = 300.0
g = 15.0
E_r = 0.0

v_reset = -1000.0
v_peak = 1000.0

# define inputs
T = 7000.0
dt = 1e-2
dts = 1e-1
cutoff = 1000.0
inp = np.zeros((int(T/dt), 1)) + cond_map[cond]["eta"]
inp[:int(300.0/dt), 0] += cond_map[cond]["eta_init"]
inp[int(2000/dt):int(5000/dt), 0] += cond_map[cond]["eta_inc"]

# define lorentzian distribution of etas
etas = eta + Delta * np.tan(0.5*np.pi*(2*np.arange(1, N+1)-N-1)/(N+1))

# define connectivity
# W = random_connectivity(N, N, 0.2)

# run the model
###############

# initialize model
node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": v_t, "eta": etas, "tau_u": tau_u, "b": b, "kappa": kappa,
             "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_r, "tau_x": tau_x}

# initialize model
net = Network(dt=dt, device="cpu")
net.add_diffeq_node("sfa", f"config/snn/adik", #weights=W, source_var="s", target_var="s_in",
                    input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                    node_vars=node_vars.copy(), op="adik_op", spike_reset=v_reset, spike_threshold=v_peak,
                    verbose=False, clear=True, N=N, float_precision="float64")

# perform simulation
obs = net.run(inputs=inp, sampling_steps=int(dts/dt), verbose=True, cutoff=int(cutoff/dt),
              record_vars=[("sfa", "u", False), ("sfa", "v", False), ("sfa", "x", False)], enable_grad=False)
s, v, u, x = (obs.to_dataframe("out"), obs.to_dataframe(("sfa", "v")), obs.to_dataframe(("sfa", "u")),
              obs.to_dataframe(("sfa", "x")))
del obs
time = s.index

# calculate the mean-field quantities
# spikes = np.zeros_like(v.values)
# n_plot = 50
# for i in range(N):
#     idx_spikes, _ = find_peaks(v.values[:, i], prominence=50.0, distance=20)
#     spikes[idx_spikes, i] = dts/dt
#     # if i % n_plot == 0:
#     #     fig, ax = plt.subplots(figsize=(12, 3))
#     #     ax.plot(v.values[:, i])
#     #     for idx in idx_spikes:
#     #         ax.axvline(x=idx, ymin=0.0, ymax=1.0, color="red")
#     #     plt.show()
spikes = s.values
r = np.mean(spikes, axis=1) / tau_s
u_widths = get_fwhm(u.values, plot_steps=10000000, jobs=10)
v_widths = get_fwhm(v.values, plot_steps=10000000, jobs=10)
u = np.mean(u.values, axis=1)
v = np.mean(v.values, axis=1)
s = np.mean(s.values, axis=1)
x = np.mean(x.values, axis=1)

# calculate the kuramoto order parameter
ko_y = v - v_r
ko_x = np.pi*C*r/k
z = (1 - ko_x + 1.0j*ko_y)/(1 + ko_x - 1.0j*ko_y)

# save results to file
pickle.dump({"results": {"spikes": spikes, "v": v, "u": u, "x": x, "r": r, "s": s, "z": 1 - np.abs(z),
                         "theta": np.imag(z), "u_width": u_widths, "v_width": v_widths}, "params": node_vars},
            open(f"results/snn_etas_{cond}.pkl", "wb"))

# plot results
fig, ax = plt.subplots(nrows=2, figsize=(12, 6))
ax[0].imshow(spikes.T, interpolation="none", cmap="Greys", aspect="auto")
ax[0].set_ylabel(r'neuron id')
ax[1].plot(time, r)
ax[1].set_ylabel(r'$r(t)$')
ax[1].set_xlabel('time')
plt.tight_layout()
plt.show()

# plot distribution dynamics
fig2, ax = plt.subplots(nrows=4, figsize=(12, 7))
ax[0].plot(time, v, color="royalblue")
ax[0].fill_between(time, v - v_widths, v + v_widths, alpha=0.3, color="royalblue", linewidth=0.0)
ax[0].set_title("v (mV)")
ax[1].plot(time, u, color="darkorange")
ax[1].fill_between(time, u - u_widths, u + u_widths, alpha=0.3, color="darkorange", linewidth=0.0)
ax[1].set_title("u (pA)")
ax[2].plot(time, 1 - np.abs(z), color="black")
ax[2].set_title("z (dimensionless)")
ax[2].set_xlabel("time (ms)")
ax[3].plot(time, np.imag(z), color="red")
ax[3].set_title("theta (dimensionless)")
ax[3].set_xlabel("time (ms)")
fig2.suptitle("SNN")
plt.tight_layout()
plt.show()
