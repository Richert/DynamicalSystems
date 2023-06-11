from rectipy import Network, circular_connectivity
import sys
cond, wdir, tdir = sys.argv[-3:]
sys.path.append(wdir)
sys.path.append("~/PycharmProjects/DynamicalSystems/reservoir_computing")
import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import pickle
from scipy.stats import rv_discrete
from scipy.ndimage import gaussian_filter1d
import h5py
import time


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


def get_signals(stim_onsets: list, cycle_steps: int, sr: int, net: Network, y0: dict, inp_indices: np.ndarray,
                sigma: float = 10.0):

    # perform simulation for each stimulation time
    signals = []
    inputs = []
    start = int(np.round(cycle_steps / sr))
    for i, stim in enumerate(stim_onsets):
        inp = np.zeros((stim + cycle_steps, N))
        inp[stim:stim + stim_width, inp_indices] = alpha
        inp = gaussian_filter1d(inp, sigma=sigma, axis=0)
        net.reset(y0)
        obs = net.run(inputs=inp, sampling_steps=sr, record_output=True, verbose=False, enable_grad=False)
        res = obs.to_numpy("out")[-start:, :]
        signals.append(gaussian_filter1d(res, sigma=sigma, axis=0).T)
        inputs.append(inp[::sr, inp_indices[0]][-start:])

    return signals, inputs


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.squeeze() / np.max(x.flatten())
    y = y.squeeze() / np.max(y.flatten())
    return float(np.mean((x - y)**2))


def get_c(X: np.ndarray, alpha: float = 1e-4):
    """
    """
    return X @ X.T + alpha*np.eye(X.shape[0])


# define parameters
###################

# sweep sizes
n_stims = 50
n_tests = 5

# working directory
# wdir = "config"
# tdir = "results"
# cond = 101

# model parameters
N = 1000
p = 0.2
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 1.0
a = 0.03
b = -2.0
d = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# device for computations
device = "cuda:0"

# sweep condition
p1 = "Delta"
p2 = "trial"

# parameter sweep definition
with open(f"{wdir}/entrainment_sweep.pkl", "rb") as f:
    sweep = pickle.load(f)
    v1s = sweep[p1]
    v2s = sweep[p2]
    f.close()
vals = [(v1, v2) for v1 in v1s for v2 in v2s]
v1, v2 = vals[int(cond)]
print(f"Condition: {p1} = {v1},  {p2} = {v2}")

# adjust parameters according to sweep condition
for param, v in zip([p1, p2], [v1, v2]):
    exec(f"{param} = {v}")

# define lorentzian of etas
thetas = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2 * v_t - v_r)

# simulation parameters
T_init = 2000.0
dt = 1e-2
sr = 10

# other analysis parameters
K_width = 100
sigma = 10
margin = 100
seq_range = 50
indices = np.arange(0, N, dtype=np.int32)
conn_pow = 0.75
gamma = 1e-3

# define connectivity
pdfs = np.asarray([dist(idx, method="inverse", zero_val=0.0, inverse_pow=conn_pow) for idx in indices])
pdfs /= np.sum(pdfs)
W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)), homogeneous_weights=False)

# condition-dependent parameters
################################

# stimulation parameters
alpha = 80.0
p_in = 0.2
freq = 4.0
T = 1e3/freq
cycle_steps = int(T/dt)
stim_onsets = np.linspace(0, T, num=n_stims+1)[:-1]
stim_phases = 2.0*np.pi*stim_onsets/T
stim_onsets = [int(onset/dt) for onset in stim_onsets]
stim_width = int(20.0/dt)
n_inputs = int(p_in*N)
center = int(N*0.5)
inp_indices = np.arange(center-int(0.5*n_inputs), center+int(0.5*n_inputs))
test_trials = list(np.arange(0, n_stims, n_tests))
train_trials = list(np.arange(0, n_stims))
for t in test_trials:
    train_trials.pop(train_trials.index(t))

# create two target signals to fit
delay = 1500
steps = int(np.round(cycle_steps / sr))
target_1 = np.zeros((steps,))
target_1[delay] = 1.0
target_1 = gaussian_filter1d(target_1, sigma=int(delay*0.1))
t = np.linspace(0, T*1e-3, steps)
f1 = 6.0
f2 = 12.0
target_2 = np.sin(2.0*np.pi*f1*t) * np.sin(2.0*np.pi*f2*t)
targets = [target_1, target_2]

###############
# simulations #
###############

etas = [45.0, 70.0]
fname = f"snn_funcgen"
f = f"{tdir}/{fname}_{cond}.h5"
hf = h5py.File(f, "w")
for i, eta in enumerate(etas):

    # initial simulation
    ####################

    # collect parameters
    node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1 / a, "b": b, "kappa": d,
                 "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

    # initialize model
    net = Network(dt=dt, device="cuda:0")
    net.add_diffeq_node("rs", node=f"{wdir}/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                        input_var="I_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                        node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike,
                        verbose=False, clear=True)

    # perform simulation to determine intrinsic oscillation frequency
    init_steps = int(T_init / dt)
    inp = np.zeros((init_steps, 1))

    # perform additional wash-out simulation to obtain a common initial state
    net.run(inputs=inp, sampling_steps=init_steps, verbose=False, enable_grad=False)
    y0 = net.state

    t0 = time.perf_counter()

    # main simulation
    #################

    # get signals for each stimulation onset
    signals, inputs = get_signals(stim_onsets, cycle_steps, sr, net, y0, inp_indices, sigma=sigma)

    # extract results for training and testing
    train_signals = [signals[idx] for idx in train_trials]
    test_signals = [signals[idx] for idx in test_trials]
    train_phases = [stim_phases[idx] for idx in train_trials]
    test_phases = [stim_phases[idx] for idx in test_trials]

    dims, cs = [], []
    for s in train_signals:

        # calculate the network dimensionality
        corr_net = np.cov(s)
        eigs = np.abs(np.linalg.eigvals(corr_net))
        dims.append(np.sum(eigs) ** 2 / np.sum(eigs ** 2))

        # calculate the network covariance matrices
        cs.append(get_c(s, alpha=gamma))

    # calculate the network kernel
    s_mean = np.mean(signals, axis=0)
    s_var = np.mean([s_i - s_mean for s_i in signals], axis=0)
    C_inv = np.linalg.inv(np.mean(cs, axis=0))
    w = C_inv @ s_mean
    K = s_mean.T @ w
    G = s_var.T @ w

    # calculate the prediction performance for concrete targets
    train_predictions = []
    test_predictions = []
    for target in targets:
        train_predictions.append(K @ target)
        w_readout = w @ target
        test_predictions.append([w_readout @ test_sig for test_sig in test_signals])

    # calculate the kernel variance
    kernel_var = np.sum(np.abs(G.flatten()))

    # calculate the kernel quality
    K_shifted = np.zeros_like(K)
    for j in range(K.shape[0]):
        K_shifted[j, :] = np.roll(K[j, :], shift=int(K.shape[1] / 2) - j)
    K_mean = np.mean(K_shifted, axis=0)
    K_var = np.var(K_shifted, axis=0)
    K_diag = np.diag(K)

    # store results
    hf = h5py.File(f, 'r+')
    g = hf.create_group(f"{i}")
    results = {"T": T, "dt": float(dt), "sr": int(sr), "p": p, "thetas": np.asarray(thetas), "eta": float(eta),
               "alpha": float(alpha), "input_indices": np.asarray(inp_indices), "dimensionality": np.mean(dims),
               "train_predictions": train_predictions, "test_predictions": test_predictions,
               "targets": targets, "train_phases": train_phases, "test_phases": test_phases,
               "kernel_variance": kernel_var, "K_mean": K_mean, "K_var": K_var, "K_diag": K_diag,
               }
    for key, val in results.items():
        g.create_dataset(key, data=val)
    hf.close()

    t1 = time.perf_counter()
    print(f"Finished job {i+1} of {len(etas)} after {t1-t0}s.")

    # # plot results
    # _, axes = plt.subplots(nrows=2, figsize=(12, 5))
    # s_all = np.concatenate(signals, axis=1)
    # inp_all = np.concatenate(inputs, axis=0)
    # s_all /= np.max(s_all)
    # inp_all /= np.max(inp_all)
    # ax = axes[0]
    # ax.plot(np.mean(s_all, axis=0), label="s")
    # ax.plot(inp_all, label="I_ext")
    # ax.legend()
    # ax.set_xlabel("time")
    # ax.set_title("Mean signal")
    # ax = axes[1]
    # im = ax.imshow(s_all, aspect="auto", interpolation="none")
    # plt.colorbar(im, ax=ax, shrink=0.4)
    # ax.set_xlabel('time')
    # ax.set_ylabel('neurons')
    # ax.set_title(f"Dim = {np.mean(dims)}")
    # plt.tight_layout()
    #
    # _, axes = plt.subplots(ncols=2, figsize=(12, 6))
    # ax = axes[0]
    # ax.plot(target_1, label="target")
    # ax.plot(train_predictions[0], label="prediction")
    # ax.set_xlabel("time")
    # ax.set_title(f"T1")
    # ax.legend()
    # ax = axes[1]
    # ax.plot(target_2, label="target")
    # ax.plot(train_predictions[1], label="prediction")
    # ax.set_xlabel("time")
    # ax.set_title(f"T2")
    # ax.legend()
    # plt.tight_layout()
    #
    # examples = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    # fig, axes = plt.subplots(nrows=len(examples), figsize=(12, 9))
    # for i, ex in enumerate(examples):
    #     ax = axes[i]
    #     ax.plot(test_predictions[ex[0]][ex[1]], label="prediction")
    #     ax.plot(targets[ex[0]], label="target")
    #     ax.legend()
    #     ax.set_xlabel("time")
    #     ax.set_ylabel("s")
    #     ax.set_title(f"Stimulation phase: {test_phases[ex[1]]}")
    # plt.tight_layout()
    # plt.show()

# save condition
g = hf.create_group("sweep")
for key, val in {p1: v1, p2: v2}.items():
    g.create_dataset(key, data=val)
hf.close()
