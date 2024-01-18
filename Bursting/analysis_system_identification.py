import numpy as np
from rectipy import Network
import pysindy as ps
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'TkAgg'


def FWHM(x, y):
    max_idx = np.argmax(y)
    half_max = np.max(y) / 2.
    left_idx = np.argmin(np.abs(y[:max_idx] - half_max)) if max_idx > 0 else 0
    right_idx = np.argmin(np.abs(y[max_idx:] - half_max)) + max_idx
    return x[right_idx] - x[left_idx]
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.axvline(x=x[left_idx], ymin=0.0, ymax=2.0*half_max, color="red")
    # ax.axvline(x=x[right_idx], ymin=0.0, ymax=2.0*half_max, color="red")
    # plt.show()


def get_fwhm(signal: np.ndarray, n_bins: int) -> np.ndarray:

    widths = []
    for n in range(signal.shape[0]):
        s = signal[n, :]
        bins = np.linspace(np.min(s), np.max(s), n_bins)
        y, _ = np.histogram(s, bins)
        x = np.asarray([(bins[i+1] + bins[i])/2.0 for i in range(n_bins-1)])
        widths.append(FWHM(x, y))
    return np.asarray(widths)


# define parameters
###################

# condition
cond = "med_sfa"
cond_map = {
    "low_sfa": {"kappa": 30.0, "eta": 100.0, "eta_inc": 30.0, "eta_init": -30.0, "b": 5.0, "delta": 5.0},
    "med_sfa": {"kappa": 100.0, "eta": 120.0, "eta_inc": 30.0, "eta_init": 0.0, "b": 5.0, "delta": 5.0},
    "high_sfa": {"kappa": 300.0, "eta": 60.0, "eta_inc": -35.0, "eta_init": 0.0, "b": -7.5, "delta": 5.0},
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
              record_vars=[("sfa", "u", False), ("sfa", "v", False), ("sfa", "x", False)])
s, v, u, x = (obs.to_dataframe("out"), obs.to_dataframe(("sfa", "v")), obs.to_dataframe(("sfa", "u")),
              obs.to_dataframe(("sfa", "x")))

# system identification
#######################

# calculate width of state variables at each time point
n_bins = 500
u_mean = np.mean(u.values, axis=1)
v_mean = np.mean(v.values, axis=1)
s_mean = np.mean(s.values, axis=1)
x_mean = np.mean(x.values, axis=1)
r_mean = s_mean/tau_s
u_widths = get_fwhm(u.values, n_bins)
v_widths = get_fwhm(v.values, n_bins)

# calculate KOP
z = 1.0 - np.real(np.abs((1 - np.pi*C*r_mean/k + 1.0j*(v_mean-v_r))/(1 + np.pi*C*r_mean/k - 1.0j*(v_mean-v_r))))

# initialize sindy model
features = ["v", "u", "s", "x", "v_var", "u_var", "z"]
X = np.stack((v_mean, u_mean, s_mean, x_mean, v_widths, u_widths, z), axis=-1)
lib = ps.PolynomialLibrary(degree=2, interaction_only=True)
opt = ps.STLSQ(threshold=0.1, max_iter=50, alpha=100.0, normalize_columns=True)
model = ps.SINDy(feature_names=features, feature_library=lib, optimizer=opt)

# fit model
model.fit(X, t=dts)

# print identified dynamical system
model.print()

# predict model widths
predictions = model.simulate(x0=X[0, :], t=np.arange(X.shape[0])*dts,
                             integrator_kws={"method": "DOP853", "rtol": 1e-8, "atol": 1e-8})

# plotting
##########

plot_features = ["v", "u", "x", "z", "u_var"]
fig, axes = plt.subplots(nrows=len(plot_features), figsize=(12, 2*len(plot_features)))
for i, f in enumerate(plot_features):

    ax = axes[i]
    idx = features.index(f)
    ax.plot(X[:, idx], label="target")
    ax.plot(predictions[:, idx], label="prediction")
    ax.set_ylabel(f)
    ax.legend()

plt.tight_layout()
plt.show()
