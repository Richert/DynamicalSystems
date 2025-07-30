from pyrates import CircuitTemplate, clear
from rectipy import Network
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
import numba as nb

print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 16.0
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['lines.linewidth'] = 1.0

# define parameters
###################

# pyrates model selection
model = "ik_eta_sfa"
op = "eta_op_sfa"

# define conditions
cond_map = {
    "no_sfa_1": {"kappa": 0.0, "eta": 10.0, "eta_inc": 50.0, "eta_init": -50.0, "b": -8.0, "delta": 0.5, "title": r"$\kappa = 0.0$, $\bar b = -5.0$, $\Delta_b = 0.5$"},
    "weak_sfa_1": {"kappa": 0.05, "eta": 30.0, "eta_inc": 50.0, "eta_init": -50.0, "b": -8.0, "delta": 0.5, "title": r"$\kappa = 0.05$, $\bar b = -5.0$, $\Delta_b = 0.5$"},
    "strong_sfa_1": {"kappa": 0.2, "eta": 60.0, "eta_inc": 50.0, "eta_init": -50.0, "b": -8.0, "delta": 0.5, "title": r"$\kappa = 0.2$, $\bar b = -5.0$, $\Delta_b = 0.5$"},
    "no_sfa_2": {"kappa": 0.0, "eta": 10.0, "eta_inc": 50.0, "eta_init": -50.0, "b": -8.0, "delta": 2.0, "title": r"$\kappa = 0.0$, $\bar b = -5.0$, $\Delta_b = 2.0$"},
    "weak_sfa_2": {"kappa": 0.05, "eta": 30.0, "eta_inc": 50.0, "eta_init": -50.0, "b": -8.0, "delta": 2.0, "title": r"$\kappa = 0.05$, $\bar b = -5.0$, $\Delta_b = 2.0$"},
    "strong_sfa_2": {"kappa": 0.2, "eta": 60.0, "eta_inc": 50.0, "eta_init": -50.0, "b": -8.0, "delta": 2.0, "title": r"$\kappa = 0.2$, $\bar b = -5.0$, $\Delta_b = 2.0$"},
}

# conditions
conditions = ["no_sfa_1", "no_sfa_2", "weak_sfa_1", "weak_sfa_2", "strong_sfa_1", "strong_sfa_2"]
for cond in conditions:

    # model parameters
    N = 1000
    C = 100.0   # unit: pF
    k = 0.7  # unit: None
    v_r = -60.0  # unit: mV
    v_t = -40.0  # unit: mV
    eta = 0.0  # unit: pA
    Delta = cond_map[cond]["delta"]
    kappa = cond_map[cond]["kappa"]
    tau_u = 50.0
    b = cond_map[cond]["b"]
    tau_s = 6.0
    tau_x = 300.0
    g = 15.0
    E_r = 0.0

    # define inputs
    T = 5500.0
    dt = 1e-2
    dts = 2e-1
    cutoff = 500.0
    inp = np.zeros((int(T/dt), 1)) + cond_map[cond]["eta"]
    inp[:int(300.0/dt), 0] += cond_map[cond]["eta_init"]
    inp[int(2000/dt):int(4000/dt), 0] += cond_map[cond]["eta_inc"]

    # define lorentzian distribution of bs and vs
    etas = np.random.permutation(eta + Delta * np.tan(0.5 * np.pi * (2 * np.arange(1, N + 1) - N - 1) / (N + 1)))
    bs = np.random.permutation(b + Delta * np.tan(0.5 * np.pi * (2 * np.arange(1, N + 1) - N - 1) / (N + 1)))
    s0 = 0.01
    vs = np.random.permutation(v_r + s0 * np.tan(0.5 * np.pi * (2 * np.arange(1, N + 1) - N - 1) / (N + 1)))
    v_reset = -2000.0
    v_peak = 2000.0

    # run the mean-field model
    ##########################

    # initialize model
    ik = CircuitTemplate.from_yaml(f"config/mf/{model}")

    # update parameters
    node_vars = {'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'Delta_b': Delta, 'kappa': kappa,
                 'tau_u': tau_u, 'b': b, 'tau_s': tau_s, 'g': g, 'E_r': E_r, 'tau_x': tau_x, 'eta': eta}
    ik.update_var(node_vars={f"p/{op}/{key}": val for key, val in node_vars.items()})

    # run simulation
    res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, cutoff=cutoff, solver='euler',
                 outputs={'s': f'p/{op}/s', 'u': f'p/{op}/u', 'v': f'p/{op}/v', 'w': f'p/{op}/w'},
                 inputs={f'p/{op}/I_ext': inp[:, 0]}, decorator=nb.njit, fastmath=True, float_precision="float64",
                 clear=False)

    # save results to file
    # pickle.dump({"results": res, "params": node_vars}, open(f"results/mf_bs_{cond}.pkl", "wb"))
    clear(ik)

    # run the SNN model
    ###################

    # initialize model
    node_vars = {"C": C, "k": k, "v_r": v_r, "v_theta": v_t, "eta": etas, "tau_u": tau_u, "b": bs, "kappa": kappa,
                 "g": g, "E_r": E_r, "tau_s": tau_s, "v": vs, "tau_x": tau_x, "s": s0}

    # initialize model
    net = Network(dt=dt, device="cpu")
    net.add_diffeq_node("sfa", f"config/snn/adik2",  # weights=W, source_var="s", target_var="s_in",
                        input_var="I_ext", output_var="s", spike_var="spike", reset_var="v", to_file=False,
                        node_vars=node_vars.copy(), op="adik_op2", spike_reset=v_reset, spike_threshold=v_peak,
                        verbose=False, clear=True, N=N, float_precision="float64")

    # perform simulation
    obs = net.run(inputs=inp, sampling_steps=int(dts / dt), verbose=False, cutoff=int(cutoff / dt),
                  record_vars=[("sfa", "u", False), ("sfa", "v", False), ("sfa", "x", False)], enable_grad=False)
    s, v, u, x = (obs.to_dataframe("out"), obs.to_dataframe(("sfa", "v")), obs.to_dataframe(("sfa", "u")),
                  obs.to_dataframe(("sfa", "x")))
    del obs
    time = s.index
    spikes = s.values
    r = np.mean(spikes, axis=1) / tau_s
    u = np.mean(u.values, axis=1)
    v = np.mean(v.values, axis=1)
    s = np.mean(s.values, axis=1)
    x = np.mean(x.values, axis=1)
    print(f"finished condition {cond}")

    # plot results
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(cond_map[cond]["title"])
    ax.plot(res.index, s, label="SNN")
    ax.plot(res.index, res["s"].values, label="FRE")
    ax.set_ylabel(r'$s(t)$')
    ax.set_xlabel("time (ms)")
    ax.legend()
    plt.tight_layout()
    fig.canvas.draw()
    # plt.savefig(f"/home/richard-gast/Documents/results/bursting_snns/simulations_{cond}.svg")
    plt.show()
