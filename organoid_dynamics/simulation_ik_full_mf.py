from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'TkAgg'
import emd
from scipy import ndimage

def generate_colored_noise(num_samples, tau, scale=1.0):
    """
    Generates Brownian noise by integrating white noise.

    Args:
        num_samples (int): The number of samples in the output Brownian noise.
        scale (float): A scaling factor for the noise amplitude.

    Returns:
        numpy.ndarray: An array containing the generated Brownian noise.
    """
    white_noise = np.random.randn(num_samples)
    x = 0.0
    colored_noise = np.zeros_like(white_noise)
    for sample in range(num_samples):# Generate white noise (Gaussian)
        x += scale*white_noise[sample] - x / tau
        colored_noise[sample] = x
    return colored_noise

# define parameters
###################

# model parameters
C = 100.0
k = 0.7
v_r = -70.0
v_t = -45.0
Delta = 1.0
eta = 85.0
b = -2.0
kappa = 5.0
U0 = 0.6
alpha = 0.0
psi = 300.0
theta = 0.02
g_a = 30.0
g_n = 0.1
g_g = 15.0
tau_w = 50.0
tau_ca = 250.0
tau_u = 100.0
tau_x = 700.0
tau_a = 5.0
tau_n = 150.0
tau_g = 10.0
tau_s = 1.0
node_vars = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'alpha': alpha,
    'g_a': g_a, 'g_n': g_n, 'g_g': g_g, 'b': b, 'U0': U0, 'tau_ca': tau_ca, 'tau_w': tau_w, 'tau_u': tau_u,
    'tau_x': tau_x, 'tau_a': tau_a, 'tau_n': tau_n, 'tau_g': tau_g, 'tau_s': tau_s, 'psi': psi, 'theta': theta
}

# define inputs
T = 10000.0
cutoff = 0.0
dt = 1e-3
dts = 1.0
s_in = 0.0
noise_lvl = 0.5
noise_tau = 50.0/dt
inp = np.zeros((int(T/dt),))
inp[int(1000/dt):int(2000/dt)] += s_in
inp += generate_colored_noise(int(T/dt), tau=noise_tau, scale=noise_lvl*np.sqrt(dt))

# run the mean-field model
##########################

# initialize model
op = "ik_full_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/pc")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in node_vars.items()})

# run simulation
res_mf = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='scipy',
                outputs={'r': f'p/{op}/r', 'w': f'p/{op}/w', 'x': f'p/{op}/x', 'ca': f'p/{op}/ca'},
                inputs={f'p/{op}/I_ext': inp}
                )
fig, axes = plt.subplots(nrows=2, figsize=(12, 5))
ax = axes[0]
ax.plot(inp)
ax.set_ylabel("I")
ax = axes[1]
ax.plot(res_mf["r"])
ax.set_ylabel("r")
ax.set_xlabel("time")

# get IMFs from firing rate fluctuations
sr = 1000
imf = emd.sift.mask_sift(res_mf["r"].values * 1e3, max_imfs=7)
IP, IF, IA = emd.spectra.frequency_transform(imf, sr, 'hilbert')
fig, ax = plt.subplots(figsize=(12, 6))
emd.plotting.plot_imfs(imf, ax=ax)
fig.suptitle("1. Sif analysis")
plt.show()

# Sift the first 5 first level IMFs
masks = np.array([25/2**ii for ii in range(8)])/sr
config = emd.sift.get_config('mask_sift')
config['max_imfs'] = 3
config['imf_opts/sd_thresh'] = 0.05
imf2 = emd.sift.mask_sift_second_layer(IA, masks, sift_args=config)
IP2, IF2, IA2 = emd.spectra.frequency_transform(imf2, sr, 'hilbert')
fig, ax = plt.subplots(figsize=(12, 6))
emd.plotting.plot_imfs(imf2, ax=ax)
fig.suptitle("2. Sif analysis")

# frequency histogram definition
carrier_hist = (1, 20, 128, 'log')
am_hist = (1e-3, 6, 64, 'log')

# Compute the 1d Hilbert-Huang transform (power over carrier frequency)
fcarrier, spec = emd.spectra.hilberthuang(IF, IA, carrier_hist, sum_imfs=False)

# Compute the 2d Hilbert-Huang transform (power over time x carrier frequency)
fcarrier, hht = emd.spectra.hilberthuang(IF, IA, carrier_hist, sum_time=False)
shht = ndimage.gaussian_filter(hht, 2)

# Compute the 3d Holospectrum transform (power over time x carrier frequency x AM frequency)
fcarrier, fam, holo = emd.spectra.holospectrum(IF, IF2, IA2, carrier_hist, am_hist)
sholo = ndimage.gaussian_filter(holo, 1)

# plot results
fig, axes = plt.subplots(nrows=5, figsize=(12, 10))
fig.suptitle("Mean-field dynamics")
ax = axes[0]
ax.plot(res_mf.index, res_mf["r"] * 1e3)
ax.set_ylabel(r'$r(t)$ (Hz)')
ax.set_title("IK (full model) Mean-Field Dynamics")
ax = axes[1]
ax.plot(res_mf.index, res_mf["w"])
ax.set_ylabel(r'$w(t)$')
ax = axes[2]
ax.plot(res_mf.index, res_mf["x"])
ax.set_ylabel(r'$x(t)$')
ax = axes[3]
ax.plot(res_mf.index, res_mf["ca"])
ax.set_ylabel(r'$ca(t)$')
ax = axes[4]
ax.plot(res_mf.index, inp[::int(dts/dt)])
ax.set_ylabel(r'$I(t)$')
ax.set_xlabel("time (ms)")
plt.tight_layout()

plt.figure(figsize=(16, 10))

# Plot a section of the time-course
plt.axes([.325, .7, .4, .25])
plt.plot(res_mf.index, res_mf["r"].values, 'k', linewidth=1)
plt.title('Original Time-series')

# Plot the 1d Hilbert-Huang Transform
plt.axes([.075, .1, .225, .5])
plt.plot(spec, fcarrier)
plt.title('1D HHT Spectrum')
plt.yscale('log')
plt.xlabel('Power')
plt.ylabel('Frequency (Hz)')
plt.yticks(2**np.arange(7), 2**np.arange(7))
plt.ylim(fcarrier[0], fcarrier[-1])
plt.xlim(0, spec.max()*1.05)

# Plot a section of the Hilbert-Huang transform
plt.axes([.325, .1, .4, .5])
plt.pcolormesh(res_mf.index, fcarrier, shht, cmap='ocean_r', shading='nearest')
plt.yscale('log')
plt.title('2-D HHT Spectrum')
plt.xlabel('Time (seconds)')
plt.yticks(2**np.arange(7), 2**np.arange(7))

# Plot a the Holospectrum
plt.axes([.75, .1, .225, .5])
plt.pcolormesh(fam, fcarrier, sholo, cmap='ocean_r', shading='nearest')
plt.yscale('log')
plt.xscale('log')
plt.title('Holospectrum')
plt.xlabel('AM Frequency (Hz)')
plt.yticks(2**np.arange(7), 2**np.arange(7))
plt.xticks([.1, .5, 1, 2, 4, 8, 16], [.1, .5, 1, 2, 4, 8, 16])

plt.show()
# save results
# pickle.dump({'mf': res_mf, "snn": res_snn}, open("results/rs_high_sfa.p", "wb"))
