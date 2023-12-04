from pyrates import CircuitTemplate, NodeTemplate
import numpy as np
import matplotlib.pyplot as plt

# parameter definition
######################

# STN neuron parameters (fitted, see script)
C_stn = 48.0   # unit: pF
k_stn = 5.0  # unit: None
v_r_stn = -56.0  # unit: mV
v_t_stn = -49.0  # unit: mV
Delta_stn = 0.5  # unit: mV
d_stn = 51.0  # unit: pA
a_stn = 0.017  # unit: 1/ms
b_stn = -5.6  # unit: nS
I_stn = 45.0  # unit: pA

# GPe-p neuron parameters
C_pro = 109.0   # unit: pF (fitted, see script)
k_pro = 4.4  # unit: None
v_r_pro = -54.0  # unit: mV
v_t_pro = -49.0  # unit: mV
Delta_pro = 1.0  # unit: mV
d_pro = 104.0  # unit: pA
a_pro = 0.213  # unit: 1/ms
b_pro = 1.8  # unit: nS
I_pro = 46.0  # unit: pA

# GPe-a neuron parameters (fitted, see script)
C_ark = 76.0   # unit: pF
k_ark = 0.7  # unit: None
v_r_ark = -52.0  # unit: mV
v_t_ark = -47.0  # unit: mV
Delta_ark = 0.6  # unit: mV
d_ark = 46.0  # unit: pA
a_ark = 0.069  # unit: 1/ms
b_ark = -4.7  # unit: nS
I_ark = 5.0  # unit: pA

# D1-SPN neuron parameters (Humphries et al. 2009)
C_d1 = 15.0   # unit: pF
k_d1 = 1.0  # unit: None
v_r_d1 = -78.0  # unit: mV
v_t_d1 = -40.0  # unit: mV
Delta_d1 = 0.5  # unit: mV
d_d1 = 67.0  # unit: pA
a_d1 = 0.01  # unit: 1/ms
b_d1 = -20.0  # unit: nS
I_d1 = 20.0  # unit: pA

# D2-SPN neuron parameters (Humphries et al. 2009)
C_d2 = 15.0   # unit: pF
k_d2 = 1.0  # unit: None
v_r_d2 = -80.0  # unit: mV
v_t_d2 = -40.0  # unit: mV
Delta_d2 = 0.5  # unit: mV
d_d2 = 91.0  # unit: pA
a_d2 = 0.01  # unit: 1/ms
b_d2 = -20.0  # unit: nS
I_d2 = 40.0  # unit: pA

# FSI neuron parameters (Humphries et al. 2009)
C_fsi = 80.0   # unit: pF
k_fsi = 1.0  # unit: None
v_r_fsi = -65.0  # unit: mV
v_t_fsi = -50.0  # unit: mV
Delta_fsi = 0.4  # unit: mV
d_fsi = 0.0  # unit: pA
a_fsi = 0.2  # unit: 1/ms
b_fsi = 0.025  # unit: nS
I_fsi = 40.0  # unit: pA

# SNR neuron parameters (fitted, see script)
C_ret = 80.0   # unit: pF
k_ret = 1.0  # unit: None
v_r_ret = -60.0  # unit: mV
v_t_ret = -55.0  # unit: mV
Delta_ret = 1.0  # unit: mV
d_ret = 0.0  # unit: pA
a_ret = 0.05  # unit: 1/ms
b_ret = 3.0  # unit: nS
I_ret = 80.0  # unit: pA

# synaptic parameters
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 6.0
tau_gaba = 12.0

# population-level coupling strengths
J = 100.0
J_stn = 0.2*J
J_pro = 0.6*J
J_ark = 0.6*J
J_d1 = 0.9*J
J_d2 = 0.9*J
J_fsi = 0.3*J
J_ret = 0.6*J

# STN inputs
J_sp = 1.0*J_stn

# GPe-p inputs
J_ps = 0.1*J_pro
J_pp = 0.1*J_pro
J_p2 = 0.8*J_pro

# GPe-a inputs
J_as = 0.05*J_ark
J_ap = 0.15*J_ark
J_a2 = 0.8*J_ark

# D1-SPN inputs
J_11 = 0.1*J_d1
J_12 = 0.2*J_d1
J_1a = 0.3*J_d1
J_1f = 0.4*J_d1

# D2-SPN inputs
J_21 = 0.1*J_d2
J_22 = 0.2*J_d2
J_2a = 0.4*J_d2
J_2f = 0.3*J_d2

# FSI inputs
J_ff = 0.8*J_fsi
J_fs = 0.2*J_fsi

# SNr inputs
J_rs = 0.1*J_ret
J_rp = 0.3*J_ret
J_r1 = 0.6*J_ret

# model initialization
######################

# initialize circuit
node = NodeTemplate.from_yaml("config/ik_mf/ik")
node_names = ["stn", "proto", "arky", "d1", "d2", "fsi", "snr"]
net = CircuitTemplate("bg", nodes={key: node for key in node_names},
                      edges=[("proto/ik_op/s", "stn/ik_op/s_i", None, {"weight": J_sp, "delay": 2.0}),
                             ("stn/ik_op/s", "proto/ik_op/s_e", None, {"weight": J_ps, "delay": 2.0}),
                             ("proto/ik_op/s", "proto/ik_op/s_i", None, {"weight": J_pp, "delay": 1.0}),
                             ("d2/ik_op/s", "proto/ik_op/s_i", None, {"weight": J_p2, "delay": 6.0}),
                             ("stn/ik_op/s", "arky/ik_op/s_e", None, {"weight": J_as, "delay": 2.0}),
                             ("proto/ik_op/s", "arky/ik_op/s_i", None, {"weight": J_ap, "delay": 1.0}),
                             ("d2/ik_op/s", "arky/ik_op/s_i", None, {"weight": J_a2, "delay": 6.0}),
                             ("d1/ik_op/s", "d1/ik_op/s_i", None, {"weight": J_11, "delay": 1.0}),
                             ("d2/ik_op/s", "d1/ik_op/s_i", None, {"weight": J_12, "delay": 1.0}),
                             ("arky/ik_op/s", "d1/ik_op/s_i", None, {"weight": J_1a, "delay": 6.0}),
                             ("fsi/ik_op/s", "d1/ik_op/s_i", None, {"weight": J_1f, "delay": 1.0}),
                             ("d1/ik_op/s", "d2/ik_op/s_i", None, {"weight": J_12, "delay": 1.0}),
                             ("d2/ik_op/s", "d2/ik_op/s_i", None, {"weight": J_22, "delay": 1.0}),
                             ("arky/ik_op/s", "d2/ik_op/s_i", None, {"weight": J_2a, "delay": 6.0}),
                             ("fsi/ik_op/s", "d2/ik_op/s_i", None, {"weight": J_2f, "delay": 1.0}),
                             ("stn/ik_op/s", "fsi/ik_op/s_e", None, {"weight": J_fs, "delay": 5.0}),
                             ("fsi/ik_op/s", "fsi/ik_op/s_i", None, {"weight": J_ff}),
                             ("stn/ik_op/s", "snr/ik_op/s_e", None, {"weight": J_rs, "delay": 2.0}),
                             ("proto/ik_op/s", "snr/ik_op/s_i", None, {"weight": J_rp, "delay": 2.0}),
                             ("d1/ik_op/s", "snr/ik_op/s_i", None, {"weight": J_r1, "delay": 6.0})
                             ])

# update parameters
stn_vars = {"C": C_stn, "k": k_stn, "v_r": v_r_stn, "v_t": v_t_stn, "eta": I_stn, "a": a_stn,
            "b": b_stn, "d": d_stn, "tau_s": tau_ampa, "v": v_r_stn, "Delta": Delta_stn, "E_e": E_ampa, "E_i": E_gaba}
pro_vars = {"C": C_pro, "k": k_pro, "v_r": v_r_pro, "v_t": v_t_pro, "eta": I_pro, "a": a_pro,
            "b": b_pro, "d": d_pro, "tau_s": tau_gaba, "v": v_r_pro, "Delta": Delta_pro, "E_e": E_ampa, "E_i": E_gaba}
ark_vars = {"C": C_ark, "k": k_ark, "v_r": v_r_ark, "v_t": v_t_ark, "eta": I_ark, "a": a_ark,
            "b": b_ark, "d": d_ark, "tau_s": tau_ampa, "v": v_r_ark, "Delta": Delta_ark, "E_e": E_ampa, "E_i": E_gaba}
d1_vars = {"C": C_d1, "k": k_d1, "v_r": v_r_d1, "v_t": v_t_d1, "eta": I_d1, "a": a_d1,
           "b": b_d1, "d": d_d1, "tau_s": tau_gaba, "v": v_r_d1, "Delta": Delta_d1, "E_e": E_ampa, "E_i": E_gaba}
d2_vars = {"C": C_d2, "k": k_d2, "v_r": v_r_d2, "v_t": v_t_d2, "eta": I_d2, "a": a_d2,
           "b": b_d2, "d": d_d2, "tau_s": tau_gaba, "v": v_r_d2, "Delta": Delta_d2, "E_e": E_ampa, "E_i": E_gaba}
fsi_vars = {"C": C_fsi, "k": k_fsi, "v_r": v_r_fsi, "v_t": v_t_fsi, "eta": I_fsi, "a": a_fsi,
            "b": b_fsi, "d": d_fsi, "tau_s": tau_gaba, "v": v_r_fsi, "Delta": Delta_fsi, "E_e": E_ampa, "E_i": E_gaba}
snr_vars = {"C": C_ret, "k": k_ret, "v_r": v_r_ret, "v_t": v_t_ret, "eta": I_ret, "a": a_ret,
            "b": b_ret, "d": d_ret, "tau_s": tau_gaba, "v": v_r_ret, "Delta": Delta_ret, "E_e": E_ampa, "E_i": E_gaba}
neuron_params = {"stn": stn_vars, "proto": pro_vars, "arky": ark_vars, "d1": d1_vars, "d2": d2_vars,
                 "fsi": fsi_vars, "snr": snr_vars}
for node in node_names:
    net.update_var(node_vars={f"{node}/ik_op/{var}": val for var, val in neuron_params[node].items()})

# model dynamics
################

# define model input
T = 2000.0
cutoff = 500.0
dt = 1e-2
dts = 1.0
start = 1000.0
stop = 1005.0
inp = np.zeros((int(T/dt),))
inp[int(start/dt):int(stop/dt)] = 350.0

# perform simulation
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, solver="heun",
              inputs={"d1/ik_op/I_ext": inp, "d2/ik_op/I_ext": inp, "stn/ik_op/I_ext": inp * 0.1},
              outputs={key: f"{key}/ik_op/r" for key in node_names},
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

# net.get_run_func(func_name="bgtc_run", file_name="bgtc", step_size=dt, backend="fortran", float_precision="float64",
#                  vectorize=False, auto=True, in_place=False, solver="scipy")
