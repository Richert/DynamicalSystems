from pyrates import CircuitTemplate, clear
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb
from scipy.interpolate import interp1d

# define parameters
###################

# condition
conditions = ["no_sfa_1", "no_sfa_2", "weak_sfa_1", "weak_sfa_2", "strong_sfa_1", "strong_sfa_2"]
model = "ik_eta_corrected"
op = "eta_op_corrected"
cond_map = {
    "no_sfa_1": {"kappa": 0.0, "eta": 0.0, "eta_inc": 30.0, "eta_init": -30.0, "b": -5.0, "delta": 5.0},
    "weak_sfa_1": {"kappa": 100.0, "eta": 0.0, "eta_inc": 35.0, "eta_init": 0.0, "b": -5.0, "delta": 5.0},
    "strong_sfa_1": {"kappa": 300.0, "eta": 0.0, "eta_inc": 50.0, "eta_init": 0.0, "b": -5.0, "delta": 5.0},
    "no_sfa_2": {"kappa": 0.0, "eta": -150.0, "eta_inc": 190.0, "eta_init": -50.0, "b": -20.0, "delta": 5.0},
    "weak_sfa_2": {"kappa": 100.0, "eta": -20.0, "eta_inc": 70.0, "eta_init": -100.0, "b": -20.0, "delta": 5.0},
    "strong_sfa_2": {"kappa": 300.0, "eta": 40.0, "eta_inc": 100.0, "eta_init": 0.0, "b": -20.0, "delta": 5.0},
}

# model parameters
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
eta = 0.0  # unit: pA
tau_u = 35.0
tau_s = 6.0
tau_x = 300.0
g = 15.0
E_r = 0.0

# mean-field simulations
########################

results = {}
for cond in conditions:

    # define condition-specific parameters
    Delta = cond_map[cond]["delta"]
    kappa = cond_map[cond]["kappa"]
    b = cond_map[cond]["b"]

    # define inputs
    T = 7000.0
    dt = 1e-2
    dts = 1e-1
    cutoff = 1000.0
    inp = np.zeros((int(T/dt),)) + cond_map[cond]["eta"]
    inp[:int(300.0/dt)] += cond_map[cond]["eta_init"]
    inp[int(2000/dt):int(5000/dt),] += cond_map[cond]["eta_inc"]

    # load ws
    data = pickle.load(open(f"results/snn_etas_{cond}_nc.pkl", "rb"))["results"]
    time = np.linspace(0, T, num=int(T/dts))
    w_interp = interp1d(time, data["u_width"], kind="cubic")
    w_in = np.zeros((int(T/dt),))
    for i, t in enumerate(np.linspace(0, T, num=int(T/dt))):
        w_in[i] = w_interp(t)

    # run the model
    ###############

    # initialize model
    ik = CircuitTemplate.from_yaml(f"config/mf/{model}")

    # update parameters
    node_vars = {'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'kappa': kappa, 'tau_u': tau_u, 'b': b,
                 'tau_s': tau_s, 'g': g, 'E_r': E_r, 'tau_x': tau_x, 'eta': eta}
    ik.update_var(node_vars={f"p/{op}/{key}": val for key, val in node_vars.items()})

    # run simulation
    res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=2*dts, cutoff=cutoff, solver='euler',
                 outputs={'v': f'p/{op}/v', 'u': f'p/{op}/u', 's': f'p/{op}/s', 'x': f'p/{op}/x'},
                 inputs={f'p/{op}/I_ext': inp, f'p/{op}/w': w_in},
                 decorator=nb.njit, fastmath=True, float_precision="float64")

    # store results
    results[cond] = res
    clear(ik)

# plotting
##########

# conditions to plot
plot_conditions = ["no_sfa", "weak_sfa", "strong_sfa"]

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# create figure layout
fig = plt.figure()
grid = fig.add_gridspec(nrows=len(plot_conditions), ncols=2)
# plt.suptitle(r"Lorentzian ansatz IV: Extracting $w(t)$ from the SNN dynamics")

for i, cond in enumerate(plot_conditions):

    for j in range(2):

        c = f"{cond}_{j+1}"

        # extract time
        time = results[c]["s"].index

        # load data
        snn_data = pickle.load(open(f"results/snn_etas_{c}.pkl", "rb"))["results"]
        mf_data = pickle.load(open(f"results/mf_etas_global_{c}.pkl", "rb"))["results"]
        mfc_data = results[c]

        # plot mean-field vs. SNN dynamics
        ax = fig.add_subplot(grid[i, j])
        ax.plot(mf_data.index, snn_data["s"], label="SNN", color="black")
        ax.plot(mf_data["s"], label="MF", color="royalblue")
        ax.plot(mfc_data["s"], label="MF-c", color="darkorange")
        if i == len(plot_conditions)-1:
            ax.set_xlabel("time (ms)")
        if j == 0 and i == 1:
            ax.set_ylabel(r"$s$ (dimensionless)")
        if i == 0 and j == 1:
            ax.legend()

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0.01, wspace=0.01)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/mf_driven.pdf')
plt.show()
