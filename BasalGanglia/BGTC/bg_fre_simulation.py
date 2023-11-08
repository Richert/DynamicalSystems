from pyrates import CircuitTemplate, NodeTemplate
import numpy as np
import matplotlib.pyplot as plt

# parameter definition
######################

# RS neuron parameters
C_e = 100.0   # unit: pF
k_e = 0.7  # unit: None
v_r_e = -60.0  # unit: mV
v_t_e = -40.0  # unit: mV
Delta_e = 1.0  # unit: mV
d_e = 100.0  # unit: pA
a_e = 0.03  # unit: 1/ms
b_e = -2.0  # unit: nS
I_e = 20.0  # unit: pA

# LTS neuron parameters
C_i = 100.0   # unit: pF
k_i = 1.0  # unit: None
v_r_i = -56.0  # unit: mV
v_t_i = -42.0  # unit: mV
Delta_i = 1.0  # unit: mV
d_i = 20.0  # unit: pA
a_i = 0.03  # unit: 1/ms
b_i = 8.0  # unit: nS
I_i = 60.0  # unit: pA

# TC neuron parameters
C_t = 200.0   # unit: pF
k_t = 1.6  # unit: None
v_r_t = -60.0  # unit: mV
v_t_t = -50.0  # unit: mV
Delta_t = 0.2  # unit: mV
d_t = 10.0  # unit: pA
a_t = 0.1  # unit: 1/ms
b_t = 15.0  # unit: nS
I_t = 200.0  # unit: pA

# TRN neuron parameters
C_r = 40.0   # unit: pF
k_r = 0.25  # unit: None
v_r_r = -65.0  # unit: mV
v_t_r = -45.0  # unit: mV
Delta_r = 0.4  # unit: mV
d_r = 50.0  # unit: pA
a_r = 0.015  # unit: 1/ms
b_r = 10.0  # unit: nS
I_r = 60.0  # unit: pA

# SPN neuron parameters
C_s = 15.0   # unit: pF
k_s = 1.0  # unit: None
v_r_s = -80.0  # unit: mV
v_t_s = -30.0  # unit: mV
Delta_s = 0.4  # unit: mV
d_s = 90.0  # unit: pA
a_s = 0.01  # unit: 1/ms
b_s = -20.0  # unit: nS
I_s = 0.0  # unit: pA

# SNR neuron parameters
C_n = 50.0   # unit: pF
k_n = 0.25  # unit: None
v_r_n = -55.0  # unit: mV
v_t_n = -44.0  # unit: mV
Delta_n = 1.0  # unit: mV
d_n = 20.0  # unit: pA
a_n = 1.0  # unit: 1/ms
b_n = 0.25  # unit: nS
I_n = 100.0  # unit: pA

# synaptic parameters
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 10.0
tau_gaba = 20.0

# population-level coupling strengths
J = 40.0
J_e = 1.0*J
J_i = 0.2*J
J_t = 0.2*J
J_r = 0.2*J
J_s = 0.5*J
J_n = 0.2*J

# RS inputs
J_ee = 0.3*J_e
J_ei = 0.4*J_e
J_et = 0.3*J_e

# LTS inputs
J_ie = 0.4*J_i
J_ii = 0.2*J_i
J_it = 0.4*J_i

# TC inputs
J_te = 0.3*J_t
J_tr = 0.3*J_t
J_tn = 0.4*J_t

# RN inputs
J_rt = 0.4*J_r
J_rr = 0.2*J_r
J_re = 0.4*J_r

# SPN inputs
J_se = 0.6*J_s
J_st = 0.4*J_s

# SNR inputs
J_ns = 1.0*J_s

# model initialization
######################

# initialize circuit
node = NodeTemplate.from_yaml("config/ik_mf/ik")
node_names = ["rs", "lts", "tc", "rn", "spn", "snr"]
net = CircuitTemplate("net", nodes={key: node for key in node_names},
                      edges=[("rs/ik_op/s", "rs/ik_op/s_e", None, {"weight": J_ee}),
                             ("rs/ik_op/s", "lts/ik_op/s_e", None, {"weight": J_ie}),
                             ("rs/ik_op/s", "tc/ik_op/s_e", None, {"weight": J_te}),
                             ("rs/ik_op/s", "rn/ik_op/s_e", None, {"weight": J_re}),
                             ("rs/ik_op/s", "spn/ik_op/s_e", None, {"weight": J_se}),
                             ("lts/ik_op/s", "rs/ik_op/s_i", None, {"weight": J_ei}),
                             ("lts/ik_op/s", "lts/ik_op/s_i", None, {"weight": J_ii}),
                             ("tc/ik_op/s", "rs/ik_op/s_e", None, {"weight": J_et}),
                             ("tc/ik_op/s", "lts/ik_op/s_e", None, {"weight": J_it}),
                             ("tc/ik_op/s", "rn/ik_op/s_e", None, {"weight": J_rt}),
                             ("tc/ik_op/s", "spn/ik_op/s_e", None, {"weight": J_st}),
                             ("rn/ik_op/s", "tc/ik_op/s_i", None, {"weight": J_tr}),
                             ("rn/ik_op/s", "rn/ik_op/s_i", None, {"weight": J_rr}),
                             ("spn/ik_op/s", "snr/ik_op/s_i", None, {"weight": J_ns, "delay": 8.0, "spread": 2.0}),
                             ("snr/ik_op/s", "tc/ik_op/s_i", None, {"weight": J_tn})
                             ])

# update parameters
pc_vars = {"C": C_e, "k": k_e, "v_r": v_r_e, "v_t": v_t_e, "eta": I_e, "a": a_e,
           "b": b_e, "d": d_e, "tau_s": tau_ampa, "v": v_t_e, "Delta": Delta_e, "E_e": E_ampa, "E_i": E_gaba}
in_vars = {"C": C_i, "k": k_i, "v_r": v_r_i, "v_t": v_t_i, "eta": I_i, "a": a_i,
           "b": b_i, "d": d_i, "tau_s": tau_gaba, "v": v_t_i, "Delta": Delta_i, "E_e": E_ampa, "E_i": E_gaba}
tc_vars = {"C": C_t, "k": k_t, "v_r": v_r_t, "v_t": v_t_t, "eta": I_t, "a": a_t,
           "b": b_t, "d": d_t, "tau_s": tau_ampa, "v": v_t_t, "Delta": Delta_t, "E_e": E_ampa, "E_i": E_gaba}
rn_vars = {"C": C_r, "k": k_r, "v_r": v_r_r, "v_t": v_t_r, "eta": I_r, "a": a_r,
           "b": b_r, "d": d_r, "tau_s": tau_gaba, "v": v_t_r, "Delta": Delta_r, "E_e": E_ampa, "E_i": E_gaba}
spn_vars = {"C": C_s, "k": k_s, "v_r": v_r_s, "v_t": v_t_s, "eta": I_s, "a": a_s,
            "b": b_s, "d": d_s, "tau_s": tau_gaba, "v": v_t_s, "Delta": Delta_s, "E_e": E_ampa, "E_i": E_gaba}
snr_vars = {"C": C_n, "k": k_n, "v_r": v_r_n, "v_t": v_t_n, "eta": I_n, "a": a_n,
            "b": b_n, "d": d_n, "tau_s": tau_gaba, "v": v_t_n, "Delta": Delta_n, "E_e": E_ampa, "E_i": E_gaba}
neuron_params = {"rs": pc_vars, "lts": in_vars, "tc": tc_vars, "rn": rn_vars, "spn": spn_vars, "snr": snr_vars}
for node in node_names:
    net.update_var(node_vars={f"{node}/ik_op/{var}": val for var, val in neuron_params[node].items()})

# model dynamics
################

# define model input
T = 3000.0
cutoff = 500.0
dt = 1e-3
dts = 1.0
start = 1000.0
stop = 2000.0
inp = np.zeros((int(T/dt),))
inp[int(start/dt):int(stop/dt)] = 100.0

# perform simulation
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, solver="scipy", method="LSODA", atol=1e-6,
              rtol=1e-6, inputs={"tc/ik_op/I_ext": inp}, outputs={key: f"{key}/ik_op/r" for key in node_names},
              clear=True, cutoff=cutoff, in_place=False)
res = res*1e3

# plotting
fig, ax = plt.subplots(figsize=(12, 4))
res.plot(ax=ax)
ax.set_xlabel("time (ms)")
ax.set_ylabel("r (Hz)")
plt.tight_layout()
plt.show()

# generate fortran run function
###############################

net.get_run_func(func_name="bgtc_run", file_name="bgtc", step_size=dt, backend="fortran", float_precision="float64",
                 vectorize=False, auto=True, in_place=False, solver="scipy")
