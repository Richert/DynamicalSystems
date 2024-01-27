from typing import Iterator

import numpy as np
from rectipy import Network
from scipy.stats import cauchy
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.special import comb
from sysidentpy.neural_network import NARXNN
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from torch import nn
from torch.nn import Parameter

plt.rcParams['backend'] = 'TkAgg'


class DNN(nn.Module):

    def __init__(self, n_in: int, n_out: int, n_hidden: tuple = (10,), activation_func: str = "tanh"):

        super().__init__()
        dims = [n_in] + list(n_hidden) + [n_out]
        self.layers = [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        if activation_func == "tanh":
            self.func = nn.Tanh()
        elif activation_func == "sigmoid":
            self.func = nn.Sigmoid()
        elif activation_func == "relu":
            self.func = nn.ReLU()
        else:
            raise ValueError("Invalid activation function. Choose better.")

    def forward(self, x):
        for layer in self.layers:
            x = self.func(layer(x))
        return x

    def parameters(self, recurse: bool = True):
        for layer in self.layers:
            yield layer.weight
            yield layer.bias


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


def smooth(signal: np.ndarray, window: int = 10):
    # N = len(signal)
    # for start in range(N):
    #     signal_window = signal[start:start+window] if N - start > window else signal[start:]
    #     signal[start] = np.mean(signal_window)
    return gaussian_filter1d(signal, sigma=window)


# define parameters
###################

# condition
cond = "high_sfa"
cond_map = {
    "low_sfa": {"kappa": 30.0, "eta": 100.0, "eta_inc": 0.0, "eta_init": -30.0, "b": 5.0, "delta": 5.0},
    "med_sfa": {"kappa": 100.0, "eta": 120.0, "eta_inc": 30.0, "eta_init": 0.0, "b": 5.0, "delta": 5.0},
    "high_sfa": {"kappa": 300.0, "eta": 60.0, "eta_inc": 0.0, "eta_init": 0.0, "b": -7.5, "delta": 5.0},
    "low_delta": {"kappa": 0.0, "eta": -125.0, "eta_inc": 135.0, "eta_init": -30.0, "b": -15.0, "delta": 1.0},
    "med_delta": {"kappa": 0.0, "eta": 100.0, "eta_inc": 30.0, "eta_init": -30.0, "b": 5.0, "delta": 5.0},
    "high_delta": {"kappa": 0.0, "eta": 6.0, "eta_inc": -40.0, "eta_init": 30.0, "b": -6.0, "delta": 10.0},
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

# run the snn model
###################

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

# system identification
#######################

# calculate the mean-field quantities
print("Starting FWHM calculation")
window = 25
r = np.zeros_like(v.values)
for i in range(N):
    spikes, _ = find_peaks(v.values[:, i], prominence=50.0, distance=20)
    r[spikes, i] = 1.0
r = smooth(np.mean(r, axis=1), window)
u_widths = smooth(get_fwhm(u.values, plot_steps=10000000, jobs=10), window)
v_widths = smooth(get_fwhm(v.values, plot_steps=10000000, jobs=10), window)
u = smooth(np.mean(u.values, axis=1), window)
v = smooth(np.mean(v.values, axis=1), window)
s = smooth(np.mean(s.values, axis=1), window)
x = smooth(np.mean(x.values, axis=1), window)

# calculate KOP
z = 1.0 - np.real(np.abs((1 - np.pi*C*r/k + 1.0j*v)/(1 + np.pi*C*r/k - 1.0j*v)))

# create input and output data
print("Creating training data")
features = ["r", "s", "v", "u", "x", "z", "|v-v_r|", "width(v)"]
X = np.stack((r, s, v, u, x, z, np.abs(v-v_r), v_widths), axis=-1)
for i in range(X.shape[1]):
    X[:, i] -= np.mean(X[:, i])
    X[:, i] /= np.std(X[:, i])
y = np.reshape(u_widths, (u_widths.shape[0], 1))

# initialize system identification model
print("Training sparse identification model")
degree = 3
n_bf = np.sum([comb(len(features), i+1, repetition=True) for i in range(degree)])
basis_functions = Polynomial(degree=degree)
net = DNN(n_in=int(n_bf), n_out=1, n_hidden=(50, 5), activation_func="tanh")
model = NARXNN(ylag=3, xlag=[[3] for _ in range(X.shape[1])], basis_function=basis_functions,
               loss_func="mse_loss", optimizer="Rprop", epochs=500, optim_params={"etas": (0.5, 1.2)},
               learning_rate=0.01, batch_size=10000, net=net, model_type="NFIR")

# fit model
model.fit(X=X, y=y)

# predict width of u
print("Model prediction")
predictions = model.predict(X=X, y=y)
rrse = root_relative_squared_error(y, predictions)

# plotting
##########

plot_features = ["r", "v", "u", "x", "z", "|v-v_r|", "width(v)"]
fig, axes = plt.subplots(nrows=len(plot_features), figsize=(12, len(plot_features)), sharex="all")
for i, f in enumerate(plot_features):
    ax = axes[i]
    idx = features.index(f)
    ax.plot(X[:, idx])
    ax.set_ylabel(f)
plt.suptitle("Predictors")
plt.tight_layout()

fig2, ax2 = plt.subplots(figsize=(12, 4), sharex="all")
ax2.plot(y[:, 0], label="target")
ax2.plot(predictions[:, 0], label="predictions")
ax2.legend()
ax2.set_xlabel("time")
ax2.set_ylabel("width(u)")
ax2.set_title(f"Squared error on test data: {rrse}")
plt.suptitle("Prediction")
plt.tight_layout()

plt.show()
