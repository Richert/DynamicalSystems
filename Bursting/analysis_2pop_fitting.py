import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import differential_evolution
from pyrates import CircuitTemplate, clear, clear_frontend_caches
from numba import njit
from typing import Union
plt.rcParams['backend'] = 'TkAgg'

model_idx = 0

# function definitions
######################


def rmse(x: np.ndarray, y: np.ndarray):
    e = np.sqrt(np.mean((x - y)**2))
    if np.isnan(e) or np.isinf(e):
        e = 1e10
    return e


def get_signal(params: np.ndarray, node_vars: dict, T: float, dt: float, dts: float, cutoff: float, inp: np.ndarray,
               maxiter: int = 10):

    # preparations
    model = "ik_2pop"
    op1 = "eta_op"
    op2 = "eta_op_c"

    # initialize model
    ik = CircuitTemplate.from_yaml(f"config/mf/{model}")
    ik.update_var(node_vars={f"p1/{op1}/{key}": val for key, val in node_vars.items()})
    ik.update_var(node_vars={f"p2/{op2}/{key}": val for key, val in node_vars.items()})
    ik.update_var(node_vars={f"p2/{op2}/eps1": params[0], f"p2/{op2}/eps2": params[1]},
                  edge_vars=[(f"p1/{op1}/s", f"p1/{op1}/s_in", {"weight": params[2]}),
                             (f"p1/{op1}/s", f"p2/{op2}/s_in", {"weight": params[2]}),
                             (f"p2/{op2}/s", f"p1/{op1}/s_in", {"weight": 1 - params[2]}),
                             (f"p2/{op2}/s", f"p2/{op2}/s_in", {"weight": 1 - params[2]})])

    # simulate model dynamics
    global model_idx
    model_idx += 1
    res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
                 outputs={'s1': f'p1/{op1}/s', 's2': f'p2/{op2}/s'},
                 inputs={f'p1/{op1}/I_ext': inp, f'p2/{op2}/I_ext': inp},
                 decorator=njit, fastmath=True, float_precision="float64", clear=False, verbose=False,
                 in_place=False, file_name=f"ik_{model_idx}")
    clear(ik)
    return res["s1"].values*params[2] + res["s2"].values*(1 - params[2])


def get_error(params: np.ndarray, signals: dict, cond_map: dict, T: float, dt: float, dts: float, cutoff: float,
              full_output: bool = False, maxiter: int = 10) -> Union[float, tuple]:

    # get predictions for each condition
    predictions = {}
    for cond in cond_map:

        # model parameters
        C = 100.0  # unit: pF
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

        # define inputs
        inp = np.zeros((int(T / dt),)) + cond_map[cond]["eta"]
        inp[:int(300.0 / dt)] += cond_map[cond]["eta_init"]
        inp[int(2000 / dt):int(5000 / dt), ] += cond_map[cond]["eta_inc"]

        # collect parameters
        node_vars = {'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'kappa': kappa, 'tau_u': tau_u, 'b': b,
                     'tau_s': tau_s, 'g': g, 'E_r': E_r, 'tau_x': tau_x, 'eta': eta}

        # get prediction
        iteration = 0
        while iteration < maxiter:
            try:
                predictions[cond] = get_signal(params, node_vars, T, dt, dts, cutoff, inp)
                break
            except (KeyError, ImportError):
                clear_frontend_caches()
                iteration += 1

    # calculate NMRSE across conditions
    error = 0.0
    for cond in conditions:
        error += rmse(signals[cond], predictions[cond])

    if full_output:
        return error, predictions
    return error


# preparations
##############

# define conditions
conditions = ["no_sfa_1", "no_sfa_2", "weak_sfa_1", "weak_sfa_2", "strong_sfa_1", "strong_sfa_2"]
cond_map = {
        "no_sfa_1": {"kappa": 0.0, "eta": 0.0, "eta_inc": 30.0, "eta_init": -30.0, "b": -5.0, "delta": 5.0},
        "weak_sfa_1": {"kappa": 100.0, "eta": 0.0, "eta_inc": 35.0, "eta_init": 0.0, "b": -5.0, "delta": 5.0},
        "strong_sfa_1": {"kappa": 300.0, "eta": 0.0, "eta_inc": 50.0, "eta_init": 0.0, "b": -5.0, "delta": 5.0},
        "no_sfa_2": {"kappa": 0.0, "eta": -150.0, "eta_inc": 190.0, "eta_init": -50.0, "b": -20.0, "delta": 5.0},
        "weak_sfa_2": {"kappa": 100.0, "eta": -20.0, "eta_inc": 70.0, "eta_init": -100.0, "b": -20.0, "delta": 5.0},
        "strong_sfa_2": {"kappa": 300.0, "eta": 40.0, "eta_inc": 100.0, "eta_init": 0.0, "b": -20.0, "delta": 5.0},
    }

# load SNN data
signals = {}
for cond in conditions:
    signals[cond] = pickle.load(open(f"results/snn_etas_{cond}.pkl", "rb"))["results"]["s"]

# simulation parameters
T = 7000.0
dt = 1e-2
dts = 2e-1
cutoff = 1000.0

# model fitting
###############

# initial parameters and boundaries
initial_coefs = np.asarray([1.0, 1.0, 0.5])
bounds = [(0.0, 3.0), (0.0, 3.0), (0.0, 1.0)]

# fitting procedure
res = differential_evolution(get_error, bounds=bounds, args=(signals, cond_map, T, dt, dts, cutoff),
                             strategy="best2bin", maxiter=100, popsize=20, tol=1e-2, atol=1e-3, disp=True, workers=16)
coefs = res.x

# get final model dynamics
error, predictions = get_error(coefs, signals, cond_map, T, dt, dts, cutoff, full_output=True)

# save results
pickle.dump({"fitted_coefs": coefs, "predictions": predictions, "error": error},
            open("results/mf_2pop_fit.pkl", "wb"))

# plotting
##########

# prepare figure
fig = plt.figure(figsize=(12, 8), dpi=130)
grid = fig.add_gridspec(nrows=3, ncols=2)
axes = []
for i in range(3):
    for j in range(2):
        axes.append(fig.add_subplot(grid[i, j]))

# plotting
for i, cond in enumerate(cond_map):

    mf_data = predictions[cond]
    snn_data = signals[cond]

    ax = axes[i]
    ax.plot(snn_data, color="black", label="snn")
    ax.plot(mf_data, color="darkorange", label="mf")
    ax.set_ylabel("u")
    if i > 3:
        ax.set_xlabel("time")
        ax.legend()

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/mf_2pop_fit.pdf')
plt.show()
