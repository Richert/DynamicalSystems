from pyrates import CircuitTemplate, clear
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb

# define parameters
###################

# pyrates model selection
model = "ik_2pop"
op1 = "eta_op"
op2 = "eta_op_c"

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

    # run simulation
    res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
                 outputs={'s1': f'p1/{op1}/s', 's2': f'p2/{op2}/s', 'u1': f'p1/{op1}/u', 'u2': f'p2/{op2}/u'},
                 inputs={f'p1/{op1}/I_ext': inp, f'p2/{op2}/I_ext': inp}, decorator=nb.njit, fastmath=True,
                 float_precision="float64", clear=False)

    # save results to file
    # pickle.dump({"results": res, "params": node_vars}, open(f"results/mf_etas_{cond}.pkl", "wb"))
    clear(ik)

    # plot results
    fig, ax = plt.subplots(nrows=4, figsize=(12, 8))
    ax[0].plot(res["s1"])
    ax[0].set_ylabel(r'$s_1(t)$')
    ax[1].plot(res["s2"])
    ax[1].set_ylabel(r'$s_2(t)$')
    ax[2].plot(res["u1"])
    ax[2].set_ylabel(r'$u_1(t)$')
    ax[3].plot(res["u2"])
    ax[3].set_ylabel(r'$u_2(t)$')
    ax[3].set_xlabel("time (ms)")
    plt.tight_layout()
    plt.show()
