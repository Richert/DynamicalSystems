from pyrates import CircuitTemplate, NodeTemplate
import numpy as np
import matplotlib.pyplot as plt

# parameter definition
######################

# STN neuron parameters
C_stn = 100.0   # unit: pF
k_stn = 0.7  # unit: None
v_r_stn = -60.0  # unit: mV
v_t_stn = -40.0  # unit: mV
Delta_stn = 1.0  # unit: mV
d_stn = 100.0  # unit: pA
a_stn = 0.03  # unit: 1/ms
b_stn = -2.0  # unit: nS
I_stn = 20.0  # unit: pA

# GPe-p neuron parameters
C_pro = 100.0   # unit: pF
k_pro = 1.0  # unit: None
v_r_pro = -56.0  # unit: mV
v_t_pro = -42.0  # unit: mV
Delta_pro = 1.0  # unit: mV
d_pro = 20.0  # unit: pA
a_pro = 0.03  # unit: 1/ms
b_pro = 8.0  # unit: nS
I_pro = 60.0  # unit: pA

# GPe-a neuron parameters
C_ark = 200.0   # unit: pF
k_ark = 1.6  # unit: None
v_r_ark = -60.0  # unit: mV
v_t_ark = -50.0  # unit: mV
Delta_ark = 0.2  # unit: mV
d_ark = 10.0  # unit: pA
a_ark = 0.1  # unit: 1/ms
b_ark = 15.0  # unit: nS
I_ark = 200.0  # unit: pA

# D1-SPN neuron parameters
C_d1 = 40.0   # unit: pF
k_d1 = 0.25  # unit: None
v_r_d1 = -65.0  # unit: mV
v_t_d1 = -45.0  # unit: mV
Delta_d1 = 0.4  # unit: mV
d_d1 = 50.0  # unit: pA
a_d1 = 0.015  # unit: 1/ms
b_d1 = 10.0  # unit: nS
I_d1 = 60.0  # unit: pA

# D2-SPN neuron parameters
C_d2 = 15.0   # unit: pF
k_d2 = 1.0  # unit: None
v_r_d2 = -80.0  # unit: mV
v_t_d2 = -30.0  # unit: mV
Delta_d2 = 0.4  # unit: mV
d_d2 = 90.0  # unit: pA
a_d2 = 0.01  # unit: 1/ms
b_d2 = -20.0  # unit: nS
I_d2 = 0.0  # unit: pA

# FSI neuron parameters
C_fsi = 50.0   # unit: pF
k_fsi = 0.25  # unit: None
v_r_fsi = -55.0  # unit: mV
v_t_fsi = -44.0  # unit: mV
Delta_fsi = 1.0  # unit: mV
d_fsi = 20.0  # unit: pA
a_fsi = 1.0  # unit: 1/ms
b_fsi = 0.25  # unit: nS
I_fsi = 100.0  # unit: pA

# synaptic parameters
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 10.0
tau_gaba = 20.0

# population-level coupling strengths
J = 40.0
J_stn = 1.0*J
J_pro = 0.2*J
J_ark = 0.2*J
J_d1 = 0.2*J
J_d2 = 0.5*J
J_fsi = 0.2*J

# STN inputs
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
