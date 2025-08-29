from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numba as nb
import pickle
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 8.0
eta = 0.0
b = 1.0
kappa = 50**2
alpha = 2.0
tau_a = 100.0
tau_u = 30.0
tau_x = 500.0
tau_s = 8.0
A0 = 0.3
g = 200.0
E_r = 0.0
I_ext = 60.0
noise_lvl = 0.0
noise_sigma = 100.0

params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'alpha': alpha,
    'tau_a': tau_a, 'tau_u': tau_u, 'g': g, 'E_r': E_r, 'b': b, 'tau_x': tau_x, 'A0': A0, 'tau_s': tau_s
}

# define inputs
T = 3000.0
cutoff = 0.0
dt = 1e-1
dts = 1.0
inp = np.zeros((int(T/dt),)) + I_ext
noise = noise_lvl*np.random.randn(inp.shape[0])
noise = gaussian_filter1d(noise, sigma=noise_sigma)
inp += noise

# define directories and file to fit
path = "/home/richard-gast/Documents"
dataset = "trujilo_2019"
load_dir = f"{path}/data/{dataset}"
n_clusters = 5
target_cluster = 0

# target data loading and processing
####################################

# load data from file
data = pickle.load(open(f"{load_dir}/{n_clusters}cluster_kmeans_results.pkl", "rb"))
waveforms = data["cluster_centroids"]
fr_target = waveforms[target_cluster]
waveform_length = len(fr_target)

# fourier transform
target_fft = np.fft.rfft(fr_target)
target_psd = np.real(np.abs(target_fft))

# run the mean-field model
##########################

# initialize model
op = "ik_ca_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/ik_ca")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in params.items()})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='heun',
                outputs={'r': f'p/{op}/r', 'a': f'p/{op}/a', 'u': f'p/{op}/u'},
                inputs={f'p/{op}/I_ext': inp}, clear=True)
fr = res_mf.loc[:, "r"].values * 1e4

# # get waveform
# max_idx_model = np.argmax(fr)
# max_idx_target = np.argmax(fr_target)
# start = max_idx_model - max_idx_target
# if start < 0:
#     start = 0
# if start + waveform_length > fr.shape[0]:
#     start = fr.shape[0] - waveform_length
# fr = fr[start:start + waveform_length]

# fourier transform
fr_fft = np.fft.rfft(fr)
fr_psd = np.real(np.abs(fr_fft))

# plotting
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(ncols=1, nrows=2)
ax = fig.add_subplot(grid[0, :])
ax.plot(fr_target, label="target")
ax.plot(fr, label="simulation")
ax.set_title("Model Dynamics")
ax.set_xlabel("time")
ax.set_ylabel("firing rate")
ax.legend()
ax = fig.add_subplot(grid[1, :])
freqs = np.fft.rfftfreq(len(fr), d=dts*1e-3)
idx = (freqs > 0.05) * (freqs <= 10.0)
ax.plot(freqs[idx], target_psd[idx], label="target")
ax.plot(freqs[idx], fr_psd[idx], label="simulation")
ax.set_xlabel("frequency")
ax.set_ylabel("power")
ax.set_title("Fourier Transform")
ax.legend()
plt.tight_layout()
plt.show()

# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
