import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def exp(t: np.ndarray, tau: float, alpha: float, beta: float):
    return alpha * np.exp(-t/tau) + beta


def get_peaks(x: np.ndarray, stim: int, prominence: float = 0.01, epsilon: float = 1e-3, **kwargs):
    peaks, _ = find_peaks(x, prominence=prominence, **kwargs)
    if len(peaks) == 1:
        x0 = x[0]
        idx = np.argwhere(x[stim:] < x0+epsilon).squeeze()[0]
        peaks = np.arange(peaks[0], idx).tolist()
    return peaks


# define parameters
###################

# general parameters
plot_examples = True

# model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
eta = 48.0
a = 0.03
b = -2.0
d = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0

# define inputs
T = 7000.0
cutoff = 2000.0
stim = 100.0 + cutoff
amp = 5000.0
sigma = 200
dt = 1e-2
dts = 1.0
inp = np.zeros((int(T/dt), 1)) + eta
inp[int(stim/dt), 0] += amp
inp[:, 0] = gaussian_filter1d(inp[:, 0], sigma=sigma)

# define sweep
deltas = np.arange(0.1, 1.5, 0.2)
taus = []

# run parameters sweep
######################

for delta in deltas:

    # initialize model
    ik = CircuitTemplate.from_yaml("config/ik_mf/ik")

    # update parameters
    ik.update_var(node_vars={'p/ik_op/C': C, 'p/ik_op/k': k, 'p/ik_op/v_r': v_r, 'p/ik_op/v_t': v_t,
                             'p/ik_op/Delta': delta, 'p/ik_op/d': d, 'p/ik_op/a': a, 'p/ik_op/b': b,
                             'p/ik_op/tau_s': tau_s, 'p/ik_op/g': g, 'p/ik_op/E_r': E_r})

    # run simulation
    res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='scipy', max_step=0.5,
                    outputs={'r': 'p/ik_op/r'}, inputs={'p/ik_op/I_ext': inp[:, 0]}, method='LSODA', rtol=1e-8, atol=1e-8)
    signal = res_mf["r"]
    time = signal.index - int(cutoff/dts)
    fr = signal.values * 1e3

    # detect decay time
    fr0 = fr[0]
    epsilon = 1e-3
    sigma_fr = 100.0
    stim_peak = int(stim/dts) - int(cutoff/dts)
    peaks = get_peaks(fr, stim_peak)
    popt, *_ = curve_fit(exp, [time[idx] for idx in peaks], [fr[idx] for idx in peaks], max_nfev=1000,
                         bounds=((0.0, 0.0, 0.0), (2000.0, 100.0, 100.0)))
    fr_exp = exp(time, *popt)
    taus.append(popt[0])

    # example plotting
    if plot_examples:
        fig, axes = plt.subplots(figsize=(12, 6), nrows=2)
        ax = axes[0]
        ax.plot(time, fr, label="raw")
        for p in peaks:
            ax.scatter(time[p], fr[p], marker="*", color="green")
        ax.plot(time, fr_exp, label="exponential fit")
        ax.set_ylabel("r (Hz)")
        ax.set_xlabel("time (ms)")
        ax.set_title(fr"Exponential fit for $\Delta_v = {delta}$ mV: $\tau = {popt[0]}$ ms.")
        ax.legend()
        ax = axes[1]
        ax.plot(time, inp[int(cutoff/dt)::int(dts/dt), 0] - eta)
        ax.set_ylabel("I_ext (pA)")
        ax.set_xlabel("time (ms)")
        ax.set_title("Input")
        plt.tight_layout()
        plt.show()

# results plotting
##################

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(deltas, taus)
ax.set_xlabel(r"$\Delta_v$ (mV)")
ax.set_ylabel(r"$\tau$ (ms)")
ax.set_title("Impulse response decay time constants")
plt.tight_layout()
plt.show()
