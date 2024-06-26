from pyrates import CircuitTemplate, clear
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb

# define parameters
###################

# pyrates model selection
model = "ik_3pop"
op1 = "eta_op"
op2 = "eta_op_c"
eps = 1.0

# define conditions
cond_map = {
        "no_sfa_1": {"kappa": 0.0, "eta": 0.0, "eta_inc": 30.0, "eta_init": -30.0, "b": -5.0, "delta": 5.0},
        "weak_sfa_1": {"kappa": 100.0, "eta": 0.0, "eta_inc": 35.0, "eta_init": 0.0, "b": -5.0, "delta": 5.0},
        "strong_sfa_1": {"kappa": 300.0, "eta": 0.0, "eta_inc": 50.0, "eta_init": 0.0, "b": -5.0, "delta": 5.0},
        "no_sfa_2": {"kappa": 0.0, "eta": -150.0, "eta_inc": 190.0, "eta_init": -50.0, "b": -20.0, "delta": 5.0},
        "weak_sfa_2": {"kappa": 100.0, "eta": -20.0, "eta_inc": 70.0, "eta_init": -100.0, "b": -20.0, "delta": 5.0},
        "strong_sfa_2": {"kappa": 300.0, "eta": 40.0, "eta_inc": 100.0, "eta_init": 0.0, "b": -20.0, "delta": 5.0},
    }

# conditions
conditions = ["no_sfa_1", "no_sfa_2", "weak_sfa_1", "weak_sfa_2", "strong_sfa_1", "strong_sfa_2"]
for cond in conditions:

    # model parameters
    C = 100.0   # unit: pF
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
    T = 7000.0
    dt = 1e-2
    dts = 2e-1
    cutoff = 1000.0
    inp = np.zeros((int(T/dt),)) + cond_map[cond]["eta"]
    inp[:int(300.0/dt)] += cond_map[cond]["eta_init"]
    inp[int(2000/dt):int(5000/dt),] += cond_map[cond]["eta_inc"]

    # run the model
    ###############

    # initialize model
    ik = CircuitTemplate.from_yaml(f"config/mf/{model}")

    # update parameters
    node_vars = {'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'kappa': kappa, 'tau_u': tau_u, 'b': b,
                 'tau_s': tau_s, 'g': g, 'E_r': E_r, 'tau_x': tau_x, 'eta': eta}
    ik.update_var(node_vars={f"p1/{op1}/{key}": val for key, val in node_vars.items()})
    ik.update_var(node_vars={f"p2/{op2}/{key}": val for key, val in node_vars.items()})
    ik.update_var(node_vars={f"p3/{op2}/{key}": val for key, val in node_vars.items()})
    ik.update_var(node_vars={f"p2/{op2}/eps": eps, f"p3/{op2}/eps": -eps})

    # run simulation
    res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
                 outputs={'s1': f'p1/{op1}/s', 's2': f'p2/{op2}/s', 's3': f'p3/{op2}/s',
                          'u1': f'p1/{op1}/u', 'u2': f'p2/{op2}/u', 'u3': f'p3/{op2}/u'},
                 inputs={f'p1/{op1}/I_ext': inp, f'p2/{op2}/I_ext': inp, f'p3/{op2}/I_ext': inp},
                 decorator=nb.njit, fastmath=True, float_precision="float64", clear=False)

    # save results to file
    # pickle.dump({"results": res, "params": node_vars}, open(f"results/mf_etas_{cond}.pkl", "wb"))
    clear(ik)

    # plot results
    fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(16, 10))

    # plot variable s
    w = [0.5, 0.25, 0.25]
    s_comb = np.zeros((res.shape[0],))
    for i in range(3):
        ax = axes[i, 0]
        sig = res[f"s{i+1}"]
        ax.plot(sig)
        ax.set_ylabel(rf"$s_{i+1}(t)$")
        s_comb += w[i]*sig
    ax = axes[3, 0]
    ax.plot(s_comb)
    ax.set_ylabel(r"$\sum s_i$")
    ax.set_xlabel("time (ms)")

    # plot variable s
    w = [0.7, 0.15, 0.15]
    u_comb = np.zeros((res.shape[0],))
    for i in range(3):
        ax = axes[i, 1]
        sig = res[f"u{i + 1}"]
        ax.plot(sig)
        ax.set_ylabel(rf"$u_{i+1}(t)$")
        u_comb += w[i] * sig
    ax = axes[3, 1]
    ax.plot(u_comb)
    ax.set_ylabel(r"$\sum u_i$")
    ax.set_xlabel("time (ms)")

    plt.tight_layout()
    plt.show()
