from pyrates import CircuitTemplate, clear
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb
from scipy.optimize import basinhopping


# function definition
#####################

def rmse(x: np.ndarray, y: np.ndarray):
    return np.sqrt(np.mean((x-y)**2))


def get_signal(w: np.ndarray, inp: np.ndarray, model: str, op: str, node_vars: dict, target_var: str):

    # initialize model
    ik = CircuitTemplate.from_yaml(f"config/mf/{model}")

    # update parameters
    ik.update_var(node_vars={f"p/{op}/{key}": val for key, val in node_vars.items()})

    # run simulation
    res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
                 outputs={target_var: f'p/{op}/{target_var}'},
                 inputs={f'p/{op}/I_ext': inp, f'p/{op}/w': w},
                 decorator=nb.njit, fastmath=True, float_precision="float64", verbose=False)
    clear(ik)

    return np.squeeze(res[target_var].values)


def fit_w(w: np.ndarray, y: np.ndarray, inp: np.ndarray, model: str, op: str, node_vars: dict, target_var: str,
          **kwargs):

    # interpolate w
    time_old = np.linspace(0, T, num=int(T / dts))
    time_new = np.linspace(0, T, num=int(T / dt))
    w_interp = np.interp(time_new, time_old, w)

    # perform simulation with current w
    y_pred = get_signal(w_interp, inp, model, op, node_vars, target_var)

    # calculate rmse
    return rmse(y, y_pred)


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

# optimization parameters
target_var = "v"
maxiter = 50
temp = 1.0
stepsize = 1.0

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

    # collect parameters
    node_vars = {'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'kappa': kappa, 'tau_u': tau_u, 'b': b,
                 'tau_s': tau_s, 'g': g, 'E_r': E_r, 'tau_x': tau_x, 'eta': eta}

    # load target data
    y = pickle.load(open(f"results/snn_etas_{cond}.pkl", "rb"))["results"][target_var]
    w0 = pickle.load(open(f"results/snn_etas_{cond}_nc.pkl", "rb"))["results"]["u_width"]

    # fit ws
    w = basinhopping(fit_w, w0, niter=maxiter, T=temp, stepsize=stepsize,
                     minimizer_kwargs={"args": (y, inp, model, op, node_vars, target_var)}).x

    results[cond] = {"w": w, "prediction": get_signal(w, inp, model, op, node_vars, target_var), "target": y}

    print(f"Condition {cond} finished.")

    # plotting
    fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
    ax = axes[0]
    ax.plot(results["target"], label="target")
    ax.plot(results["prediction"], label="prediction")
    ax.legend()
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(target_var)
    ax = axes[1]
    ax.plot(w)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("w")
    plt.tight_layout()
    plt.show()

# store results
pickle.dump(results, open("results/w_fitting.pkl", "wb"))
