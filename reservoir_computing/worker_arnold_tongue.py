import sys
cond, wdir, tdir = sys.argv[-3:]
sys.path.append(wdir)
from rectipy import Network, random_connectivity, circular_connectivity
import numpy as np
import pickle
from scipy.stats import rv_discrete
from utility_funcs import lorentzian, get_dim, dist, butter_bandpass_filter, analytic_signal, coherence
from pandas import DataFrame


# parameters and preparations
#############################

# working directory
# wdir = "config"
# tdir = "results"

# sweep condition
# cond = 0
p1 = "p_in"
p2 = "alpha"

# network parameters
N = 1000
p = 0.1
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Deltas = [1.0, 0.1]
eta = 55.0
a = 0.03
b = -2.0
d = 100.0
g = 15.0
E_r = 0.0
tau_s = 6.0
v_spike = 1000.0
v_reset = -1000.0

# parameter sweep definition
with open(f"{wdir}/arnold_tongue_sweep.pkl", "rb") as f:
    sweep = pickle.load(f)
    v1s = sweep[p1]
    v2s = sweep[p2]
    f.close()
vals = [(v1, v2) for v1 in v1s for v2 in v2s]
v1, v2 = vals[int(cond)]

# simulation parameters
cutoff = 10000.0
T = 200000.0 + cutoff
dt = 1e-2
sr = 100
steps = int(np.round(T/dt))
time = np.linspace(0.0, T, num=steps)
fs = int(np.round(1e3/(dt*sr), decimals=0))

# input definition
omega = 5.0
alpha = 1.0
p_in = 0.1
I_ext = np.zeros((steps, 1))
I_ext[:, 0] = np.sin(2.0*np.pi*omega*time*1e-3)
ko = I_ext[::sr, 0]

# filtering options
print(f"Sampling frequency: {fs}")
f_margin = 0.5
print(f"Frequency band width (Hz): {2*f_margin}")
f_order = 6
f_cutoff = int(np.round(cutoff/(dt*sr), decimals=0))

# simulation
############

# prepare results storage
results = {"sweep": {p1: v1, p2: v2}, "T": T, "dt": dt, "sr": sr, "omega": omega, "p": p, "Deltas": Deltas}
n_reps = 5
res_cols = ["Delta", "coh_inp", "coh_noinp", "dim"]
entrainment = DataFrame(np.zeros((n_reps, len(res_cols))), columns=res_cols)
covariances = {"Delta": [], "cov": []}

# loop over repetitions
i = 0
for Delta in Deltas:
    for _ in range(n_reps):

        # simulation preparations
        #########################

        # adjust parameters according to sweep condition
        for param, v in zip([p1, p2], [v1, v2]):
            exec(f"{param} = {v}")

        # create connectivity matrix
        connectivity = "random"
        if connectivity == "circular":
            indices = np.arange(0, N, dtype=np.int32)
            pdfs = np.asarray([dist(idx, method="inverse_squared") for idx in indices])
            pdfs /= np.sum(pdfs)
            W = circular_connectivity(N, p, spatial_distribution=rv_discrete(values=(indices, pdfs)))
        else:
            W = random_connectivity(N, N, p, normalize=True)

        # create input matrix
        W_in = np.zeros((N, 1))
        idx = np.random.choice(np.arange(N), size=int(N*p_in), replace=False)
        W_in[idx, 0] = alpha

        # create background current distribution
        thetas = lorentzian(N, v_t, Delta, v_r, 2 * v_t - v_r)

        # collect remaining model parameters
        node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": thetas, "eta": eta, "tau_u": 1/a, "b": b, "kappa": d, "g": g,
                     "E_r": E_r, "tau_s": tau_s, "v": v_t}

        # initialize model
        net = Network.from_yaml(f"{wdir}/ik/rs", weights=W, source_var="s", target_var="s_in",
                                input_var="s_ext", output_var="s", spike_var="spike", spike_def="v", to_file=False,
                                node_vars=node_vars.copy(), op="rs_op", spike_reset=v_reset, spike_threshold=v_spike, dt=dt,
                                verbose=False, clear=True, device="cuda:0")
        net.add_input_layer(1, weights=W_in)

        # simulation
        ############

        obs = net.run(inputs=I_ext, sampling_steps=sr, record_output=True, verbose=False)
        ik_net = obs["out"]

        # compute entrainment and other variables of interest
        #####################################################

        ik_inp = np.mean(ik_net.loc[:, W_in[:, 0] > 0].values, axis=-1)
        ik_noinp = np.mean(ik_net.loc[:, W_in[:, 0] < alpha].values, axis=-1)

        # calculate coherence and PLV
        for ik, c in zip([ik_inp, ik_noinp], ["inp", "noinp"]):

            # scale data
            ik -= np.min(ik)
            ik_max = np.max(ik)
            if ik_max > 0.0:
                ik /= ik_max

            # filter data around driving frequency
            ik_filtered = butter_bandpass_filter(ik, (omega - f_margin * omega, omega + f_margin * omega), fs=fs,
                                                 order=f_order)
            ik_max = np.max(ik_filtered)
            if ik_max > 0:
                ik_filtered /= ik_max

            # get analytic signals
            ik_phase, ik_env = analytic_signal(ik_filtered[f_cutoff:-f_cutoff])
            ko_phase, ko_env = analytic_signal(ko[f_cutoff:-f_cutoff])

            # calculate coherence
            coh = coherence(ik_phase, ko_phase, ik_env, ko_env)

            # test plotting
            # plt.figure(2)
            # plt.plot(ik_filtered, label="ik_f")
            # plt.plot(ko, label="ko")
            # plt.plot(ik, label="ik")
            # plt.title(f"Coh = {coh}, PLV = {plv}")
            # plt.legend()
            # plt.show()

            # store results
            entrainment.loc[i, f"coh_{c}"] = coh

        # calculate dimensionality of network dynamics
        dim, cov = get_dim(ik_net.values)
        entrainment.loc[i, "dim"] = dim
        entrainment.loc[i, "Delta"] = Delta
        covariances["Delta"].append(Delta)
        covariances["cov"].append(cov)

        # go to next run
        i += 1
        print(f"Run {i} done for condition {cond}.")

# save results
fname = f"rs_entrainment"
results["cov"] = covariances
results["entrainment"] = entrainment
with open(f"{tdir}/{fname}_{cond}.pkl", "wb") as f:
    pickle.dump(results, f)
    f.close()
