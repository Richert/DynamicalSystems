import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from rectipy import Network
from pyrates import CircuitTemplate
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
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
              record_vars=[("sfa", "u", False), ("sfa", "v", False), ("sfa", "x", False)])
s, v, u, x = (obs.to_dataframe("out"), obs.to_dataframe(("sfa", "v")), obs.to_dataframe(("sfa", "u")),
              obs.to_dataframe(("sfa", "x")))

# run the mean-field model
##########################

model = "ik_eta_6d"
op = "eta_op_6d"

# initialize model
ik = CircuitTemplate.from_yaml(f"config/mf/{model}")

# update parameters
node_vars = {'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'kappa': kappa, 'tau_u': tau_u, 'b': b,
             'tau_s': tau_s, 'g': g, 'E_r': E_r, 'tau_x': tau_x, 'eta': eta}
ik.update_var(node_vars={f"p/{op}/{key}": val for key, val in node_vars.items()})

# run simulation
res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
             outputs={'s': f'p/{op}/s', 'u': f'p/{op}/u', 'v': f'p/{op}/v', 'r': f'p/{op}/r',
                      'x': f'p/{op}/x', 'w': f'p/{op}/w'},
             inputs={f'p/{op}/I_ext': inp}, decorator=nb.njit, fastmath=True, float_precision="float64")

# collect results
r_mf = res["r"].values
v_mf = res["v"].values
u_mf = res["u"].values
w_mf = res["w"].values
x_mf = res["x"].values
s_mf = res["s"].values

# FWHM analysis
###############

# calculate width of state variables at each time point
n_bins = 500
u_mean = np.mean(u.values, axis=1)
v_mean = np.mean(v.values, axis=1)
s_mean = np.mean(s.values, axis=1)
x_mean = np.mean(x.values, axis=1)
u_widths = get_fwhm(u.values, n_bins)
v_widths = get_fwhm(v.values, n_bins)

# fit the width of u via mean-field variables
r_mean = s_mean/tau_s
z = np.abs((1 - np.pi*C*r_mean/k + 1.0j*(v_mean-v_r))/(1 + np.pi*C*r_mean/k - 1.0j*(v_mean-v_r)))
predictors = [u_mean, v_mean, r_mean, x_mean, v_mean-v_r, np.abs(v_mean-v_r),
              z, b*(np.pi*C*r_mean/k)**(1-z), b*(np.pi*C*r_mean/k)**z, b*(np.pi*C*r_mean/k)*(1-z),
              b*(np.pi*C*r_mean/k)*z]
predictor_names = ["u", "v", "r", "x", "v-v-r", "|v-v_r|", "z", "w^(1-z))", "w^z", "w*(1-z))", "w*z"]
predictors = np.asarray(predictors).T
glm = Lasso()
glm.fit(predictors, u_widths)
coefs = glm.coef_
u_width_fitted = glm.predict(predictors)

# plotting
##########

# plot spiking dynamics
fig, ax = plt.subplots(nrows=2, figsize=(12, 6))
spikes = s.values
time = s.index
ax[0].imshow(spikes.T, interpolation="none", cmap="Greys", aspect="auto")
ax[0].set_ylabel(r'neuron id')
ax[1].plot(time, s_mean, label="SNN", color="black")
ax[1].plot(time, res["s"].values, label="MF", color="darkorange")
ax[1].set_ylabel(r'$s(t)$')
ax[1].set_xlabel('time')
ax[1].legend()
plt.tight_layout()

# plot distribution dynamics for SNN
fig2, ax2 = plt.subplots(nrows=3, figsize=(12, 6))
ax2[0].plot(time, v_mean, color="royalblue")
ax2[0].fill_between(time, v_mean - v_widths, v_mean + v_widths, alpha=0.3, color="royalblue", linewidth=0.0)
ax2[0].set_title("v (mV)")
ax2[1].plot(time, u_mean, color="darkorange")
ax2[1].fill_between(time, u_mean - u_widths, u_mean + u_widths, alpha=0.3, color="darkorange", linewidth=0.0)
ax2[1].set_title("u (pA)")
ax2[2].plot(time, s_mean, color="black")
ax2[2].set_title("s (dimensionless)")
ax2[2].set_xlabel("time (ms)")
fig2.suptitle("SNN")
plt.tight_layout()

# plot distribution dynamics for MF
u_delta = np.abs(b*(v_mf-v_r))/np.pi**2 - u_mf/Delta
v_delta = np.pi*C*r_mf/k
fig3, ax3 = plt.subplots(nrows=3, figsize=(12, 6))
ax3[0].plot(time, v_mf, color="royalblue")
ax3[0].fill_between(time, v_mf - v_delta, v_mf + v_delta, alpha=0.3, color="royalblue", linewidth=0.0)
ax3[0].set_title("v (mV)")
ax3[1].plot(time, u_mf, color="darkorange")
ax3[1].fill_between(time, u_mf - u_delta, u_mf + u_delta, alpha=0.3, color="darkorange", linewidth=0.0)
ax3[1].set_title("u (pA)")
ax3[2].plot(time, s_mf, color="black")
ax3[2].set_title("s (dimensionless)")
ax3[2].set_xlabel("time (ms)")
fig3.suptitle("MF")
plt.tight_layout()

# plot the mean-field fit:
print(f"Fitted coefficients: {[f'{key}: {val}' for key, val in zip(predictor_names, coefs)]}")
fig4, ax4 = plt.subplots(figsize=(12, 5), nrows=2)
ax4[0].plot(time, u_mean, color="darkorange")
ax4[0].fill_between(time, u_mean - u_widths, u_mean + u_widths, alpha=0.3, color="darkorange")
ax4[0].set_title("Target")
ax4[1].plot(time, u_mean, color="darkorange")
ax4[1].fill_between(time, u_mean - u_width_fitted, u_mean + u_width_fitted, alpha=0.3, color="darkorange")
ax4[1].set_title("Fit")
plt.tight_layout()

plt.show()
