import matplotlib.pyplot as plt
import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, grid_search
# extract phases from signals
from scipy.signal import hilbert, butter, sosfilt, coherence
from numba import njit


def get_phase(signal, N, freqs, fs):
    filt = butter(N, freqs, output="sos", btype='bandpass', fs=fs)
    s_filtered = sosfilt(filt, signal)
    return np.unwrap(np.angle(hilbert(s_filtered)))


# network definition
####################

# define network nodes
ko = NodeTemplate.from_yaml("model_templates.oscillators.kuramoto.sin_pop")
ik = NodeTemplate.from_yaml("model_templates.neural_mass_models.ik.ik_theta_pop")
nodes = {'ik': ik, 'ko': ko}

# define network edges
edges = [
    ('ko/sin_op/s', 'ik/ik_theta_op/r_in', None, {'weight': 0.001}),
    ('ik/ik_theta_op/r', 'ik/ik_theta_op/r_in', None, {'weight': 1.0})
]

# initialize network
net = CircuitTemplate(name="ik_forced", nodes=nodes, edges=edges)

# update izhikevich parameters
node_vars = {
    "C": 100.0,
    "k": 0.7,
    "v_r": -60.0,
    "v_t": -40.0,
    #"eta": 55.0,
    "Delta": 1.0,
    "g": 15.0,
    "E_r": 0.0,
    "b": -2.0,
    "a": 0.03,
    "d": 100.0,
    "tau_s": 6.0,
}
node, op = "ik", "ik_theta_op"
net.update_var(node_vars={f"{node}/{op}/{var}": val for var, val in node_vars.items()})

# perform parameter sweep
#########################

# define sweep
alphas = np.asarray([0.0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064])
omegas = np.linspace(-0.003, 0.003, 7) + 0.004
sweep = {"alpha": alphas, "omega": omegas}
param_map = {"alpha": {"vars": ["weight"], "edges": [('ko/sin_op/s', 'ik/ik_theta_op/r_in')]},
             "omega": {"vars": ["phase_op/omega"], "nodes": ["ko"]}}

# simulation parameters
T = 10000.0
dt = 1e-3
dts = 1e-2
inp = np.zeros((int(T/dt),)) + 55.0

# perform sweep
res, res_map = grid_search(net, param_grid=sweep, param_map=param_map, simulation_time=T, step_size=dt,
                           solver="scipy", method="DOP853", atol=1e-5, rtol=1e-4, sampling_step_size=dts,
                           permute_grid=True, vectorize=True, inputs={"ik/ik_theta_op/I_ext": inp},
                           outputs={"ik": "ik/ik_theta_op/r", "ko": "ko/phase_op/theta"}, decorator=njit
                           )

# coherence calculation
#######################

# calculate and store coherences
coherences = np.zeros((len(alphas), len(omegas)))
nps = 1024
window = 'hamming'
for key in res_map.index:

    # extract parameter set
    omega = res_map.at[key, 'omega']
    alpha = res_map.at[key, 'alpha']

    # collect phases
    p1 = np.sin(get_phase(res['ik'][key].squeeze(), N=10,
                          freqs=(omega-0.3*omega, omega+0.3*omega), fs=1/dts))
    p2 = np.sin(2 * np.pi * res['ko'][key].squeeze())

    # calculate coherence
    freq, coh = coherence(p1, p2, fs=1/dts, nperseg=nps, window=window)

    # find coherence matrix position that corresponds to these parameters
    idx_r = np.argmin(np.abs(alphas - alpha))
    idx_c = np.argmin(np.abs(omegas - omega))

    # store coherence value at driving frequency
    tf = freq[np.argmin(np.abs(freq - omega))]
    coherences[idx_r, idx_c] = np.max(coh[(freq >= tf-0.3*tf) * (freq <= tf+0.3*tf)])

# plot the coherence at the driving frequency for each pair of omega and J
fix, ax = plt.subplots(figsize=(12, 8))
cax = ax.imshow(coherences[::-1, :], aspect='equal')
ax.set_xlabel(r'$\omega$ (Hz)')
ax.set_ylabel(r'$J\alpha$ (Hz)')
ax.set_xticks(np.arange(0, len(alphas), 3))
ax.set_yticks(np.arange(0, len(omegas), 3))
ax.set_xticklabels(np.round(omegas[::3]*1e3, decimals=0))
ax.set_yticklabels(np.round(alphas[::-3]*1e3, decimals=0))
plt.title("Coherence between IK population and KO")
plt.colorbar(cax)
plt.show()
