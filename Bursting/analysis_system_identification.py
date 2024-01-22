import numpy as np
from rectipy import Network
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.stats import cauchy
from joblib import Parallel, delayed
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
T = 10000.0
dt = 1e-2
dts = 1e-1
cutoff = 1000.0
inp = np.zeros((int((T + cutoff)/dt), 1)) + cond_map[cond]["eta"]

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

# calculate the mean-field quantities
print("Starting FWHM calculation")
u_mean = np.mean(u.values, axis=1)
v_mean = np.mean(v.values, axis=1)
s_mean = np.mean(s.values, axis=1)
x_mean = np.mean(x.values, axis=1)
r_mean = s_mean/tau_s
u_widths = get_fwhm(u.values, plot_steps=10000000, jobs=10)

# calculate KOP
z = 1.0 - np.real(np.abs((1 - np.pi*C*r_mean/k + 1.0j*(v_mean-v_r))/(1 + np.pi*C*r_mean/k - 1.0j*(v_mean-v_r))))

# create input and output data
print("Generating training data")
features = ["r", "v", "u", "x", "width(u)"]
X = np.stack((r_mean, v_mean, u_mean, x_mean, u_widths), axis=-1)
for i in range(X.shape[1]):
    X[:, i] -= np.mean(X[:, i])
    # X[:, i] /= np.std(X[:, i])

# initialize sindy model
lib = ps.PolynomialLibrary(degree=3, interaction_only=False)
opt = ps.FROLS(max_iter=5, alpha=10.0, verbose=False)
model = ps.SINDy(feature_names=features, feature_library=lib, optimizer=opt)

# fit model
print("Fitting the model")
model.fit(X, t=dts)

# print identified dynamical system
model.print()

# predict model widths
print("Generating model predictions")
predictions = model.simulate(x0=X[0, :], t=np.arange(X.shape[0])*dts, integrator="solve_ivp",
                             integrator_kws={"method": "RK45", "rtol": 1e-7, "atol": 1e-7,
                                             "max_step": dts, "min_step": 1e-5})

# plotting
##########

plot_features = ["r", "v", "u", "x", "width(u)"]
fig, axes = plt.subplots(nrows=len(plot_features), figsize=(12, 2*len(plot_features)))
for i, f in enumerate(plot_features):

    ax = axes[i]
    idx = features.index(f)
    ax.plot(X[:, idx], label="target")
    ax.plot(predictions[:, idx], label="prediction")
    ax.set_ylabel(f)
    ax.legend()
    ymin, ymax = np.min(X[:, idx]), np.max(X[:, idx])
    ax.set_ylim([ymin-0.2*np.abs(ymin), ymax+0.2*np.abs(ymax)])

plt.tight_layout()
plt.show()
