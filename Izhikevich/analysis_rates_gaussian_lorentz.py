import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import pickle
from scipy.stats import cauchy, norm
from rectipy import Network, random_connectivity


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def gaussian(n, mu: float, sd: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = norm.rvs(loc=mu, scale=sd)
        while s <= lb or s >= ub:
            s = norm.rvs(loc=mu, scale=sd)
        samples[i] = s
    return samples


def get_fr(inp: np.ndarray, k: float, C: float, v_reset: float, v_spike: float, v_r: float, v_t: float):
    fr = np.zeros_like(inp)
    alpha = v_r+v_t
    mu = 4*(v_r*v_t + inp/k) - alpha**2
    idx = mu > 0
    mu_sqrt = np.sqrt(mu[idx])
    fr[idx] = k*mu_sqrt/(2*C*(np.arctan((2*v_spike-alpha)/mu_sqrt) - np.arctan((2*v_reset-alpha)/mu_sqrt)))
    return fr


##############
# parameters #
##############

# load data
mapping = pickle.load(open("results/norm_lorentz_fit.pkl", "rb"))

# choose neuron type
neuron_type = "rs"

if neuron_type == "rs":

    C = 100.0
    k = 0.7
    v_r = -60.0
    v_t = -40.0
    eta = 50.0
    a = 0.03
    b = -2.0
    d = 10.0
    g = 15.0
    E_r = 0.0
    tau_s = 6.0

elif neuron_type == "fs":

    C = 20.0
    k = 1.0
    v_r = -55.0
    v_t = -40.0
    eta = 70.0
    a = 0.2
    b = 0.025
    d = 0.0
    g = 5.0
    E_r = -65.0
    tau_s = 8.0

elif neuron_type == "lts":

    C = 10.0
    k = 1.0
    v_r = -56.0
    v_t = -42.0
    eta = 150.0
    a = 0.03
    b = 8.0
    d = 20.0
    g = 5.0
    E_r = -65.0
    tau_s = 8.0

else:

    raise ValueError("Wrong neuron type")

# network parameters
I_ext = np.linspace(50.0, 100.0, num=100)
v_reset = -1000.0
v_spike = 1000.0
N = 1000
N2 = 100000
p = 0.2
SDs = [2.0, 4.0]

# find deltas corresponding to SDs
Deltas = []
for SD in SDs:
    idx = np.argmin(np.abs(np.asarray(mapping["norm"]) - SD))
    Deltas.append(np.round(mapping["lorentz"][idx], decimals=1))

# time simulation parameters
dt = 1e-2
dts = 1e-1
T = 1000.0

# other parameters
n_bins = 50

# results
results = {"lorentz": {"rate_dist": [], "spikes": []}, "gauss": {"rate_dist": [], "spikes": []}, "Deltas": Deltas,
           "SDs": SDs}

############
# analysis #
############

for Delta, SD in zip(Deltas, SDs):

    # define lorentzian of etas
    thetas_l = lorentzian(N2, eta=v_t, delta=Delta, lb=v_r, ub=2*v_t-v_r)
    thetas_g = gaussian(N2, mu=v_t, sd=SD, lb=v_r, ub=2*v_t-v_r)

    for key, thetas in zip(["lorentz", "gauss"], [thetas_l, thetas_g]):

        # get firing rates of each neuron
        frs = []
        for theta in thetas:
            frs.append(get_fr(I_ext, k, C, v_reset, v_spike, v_r, theta))
        frs = np.asarray(frs)*1e3

        # get firing rate distributions for each input strength
        rate_bins = np.linspace(np.min(frs.flatten()), np.max(frs.flatten()), n_bins+1)
        densities = []
        for idx in range(len(I_ext)):
            ps, _ = np.histogram(frs[:, idx], rate_bins, density=False)
            densities.append(ps / np.sum(ps))

        # store results
        results[key]["rate_dist"].append(np.asarray(densities))

    ###########################
    # spiking data simulation #
    ###########################

    # define input
    steps = int(T/dt)
    inp = np.zeros((steps, N))

    # define lorentzian of etas
    thetas_l = lorentzian(N, eta=v_t, delta=Delta, lb=v_r, ub=2 * v_t - v_r)
    thetas_g = gaussian(N, mu=v_t, sd=SD, lb=v_r, ub=2 * v_t - v_r)

    # get connectivity
    W = random_connectivity(N, N, p, normalize=True)

    for key, thetas in zip(["lorentz", "gauss"], [thetas_l, thetas_g]):

        # collect parameters
        node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d,
                     "g": g, "E_r": E_r, "tau_s": tau_s, "v": v_t}

        # initialize network
        net = Network(dt=dt, device="cuda:0")
        net.add_diffeq_node("ik", "config/ik_snn/rs", weights=W, source_var="s", target_var="s_in",
                            input_var="I_ext", output_var="s", spike_var="spike", spike_def="v",
                            node_vars=node_vars, op="rs_op", spike_reset=v_reset, spike_threshold=v_spike)

        # run simulation
        obs = net.run(inp, sampling_steps=int(dts/dt), record_output=True, verbose=False, enable_grad=False)

        # store results
        results[key]["spikes"].append(obs.to_numpy("out"))

# save results
pickle.dump(results, open(f"results/norm_lorentz_{neuron_type}.pkl", "wb"))

############
# plotting #
############

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# create figure with grid
fig = plt.figure(figsize=(12, 5))
grid = fig.add_gridspec(nrows=2, ncols=6)

# plot the firing rate distributions for the Lorentzian
for idx, Delta, rate_dist in zip([0, 3], Deltas, results["lorentz"]["rate_dist"]):

    ax = fig.add_subplot(grid[0, idx])
    im = ax.imshow(np.asarray(rate_dist).T, cmap=plt.get_cmap('cividis'), aspect='auto', vmin=0.0, vmax=1.0,
                   origin='lower')
    ax.set_xlabel(r"$I$ (pA)")
    ax.set_ylabel(r"$r_i$")
    ax.set_title(fr"$\Delta_v = {Delta}$")
    plt.colorbar(im, ax=ax, shrink=0.8)

# plot the firing rate distributions for the Gaussian
for idx, SD, rate_dist in zip([0, 3], SDs, results["gauss"]["rate_dist"]):
    ax = fig.add_subplot(grid[1, idx])
    im = ax.imshow(np.asarray(rate_dist).T, cmap=plt.get_cmap('cividis'), aspect='auto', vmin=0.0, vmax=1.0,
                   origin='lower')
    ax.set_xlabel(r"$I$ (pA)")
    ax.set_ylabel(r"$r_i$")
    ax.set_title(fr"$\sigma_v = {SD}$")
    plt.colorbar(im, ax=ax, shrink=0.8)

# spiking raster plots for the Lorentzian
for idx, Delta, spikes in zip([(1, 3), (4, 6)], Deltas, results["lorentz"]["spikes"]):
    ax = fig.add_subplot(grid[0, idx[0]:idx[1]])
    ax.imshow(spikes.T, interpolation="none", aspect="auto", cmap="Greys")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron id")
    ax.set_title(fr"$\Delta_v = {Delta}$")

# spiking raster plots for the Gaussian
for idx, SD, spikes in zip([(1, 3), (4, 6)], SDs, results["gauss"]["spikes"]):
    ax = fig.add_subplot(grid[1, idx[0]:idx[1]])
    ax.imshow(spikes.T, interpolation="none", aspect="auto", cmap="Greys")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron id")
    ax.set_title(fr"$\sigma_v = {SD}$")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.show()
