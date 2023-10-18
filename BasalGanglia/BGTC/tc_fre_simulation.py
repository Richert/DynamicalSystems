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
Delta_e = 0.5  # unit: mV
d_e = 100.0  # unit: pA
a_e = 0.03  # unit: 1/ms
b_e = -2.0  # unit: nS
I_e = 80.0  # unit: pA

# LTS neuron parameters
C_i = 100.0   # unit: pF
k_i = 1.0  # unit: None
v_r_i = -56.0  # unit: mV
v_t_i = -42.0  # unit: mV
Delta_i = 1.0  # unit: mV
d_i = 20.0  # unit: pA
a_i = 0.03  # unit: 1/ms
b_i = 8.0  # unit: nS
I_i = 40.0  # unit: pA

# Tha neuron parameters
C_t = 200.0   # unit: pF
k_t = 1.6  # unit: None
v_r_t = -60.0  # unit: mV
v_t_t = -50.0  # unit: mV
Delta_t = 0.1  # unit: mV
d_t = 100.0  # unit: pA
a_t = 0.1  # unit: 1/ms
b_t = 15.0  # unit: nS
I_t = 100.0  # unit: pA

# synaptic parameters
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 10.0
tau_gaba = 20.0
k_ee = 10.0
k_ei = 5.0
k_et = 10.0
k_ie = 5.0
k_ii = 5.0
k_it = 5.0
k_te = 5.0

# model initialization
######################

# initialize circuit
node = NodeTemplate.from_yaml("config/ik_mf/ik")
node_names = ["rs", "lts", "tc"]
net = CircuitTemplate("net", nodes={key: node for key in node_names},
                      edges=[("rs/ik_op/s", "rs/ik_op/s_e", None, {"weight": k_ee}),
                             ("rs/ik_op/s", "lts/ik_op/s_e", None, {"weight": k_ie}),
                             ("rs/ik_op/s", "tc/ik_op/s_e", None, {"weight": k_te}),
                             ("lts/ik_op/s", "rs/ik_op/s_i", None, {"weight": k_ei}),
                             ("lts/ik_op/s", "lts/ik_op/s_i", None, {"weight": k_ii}),
                             ("tc/ik_op/s", "rs/ik_op/s_e", None, {"weight": k_et}),
                             ("tc/ik_op/s", "lts/ik_op/s_e", None, {"weight": k_it})])

# update parameters
pc_vars = {"C": C_e, "k": k_e, "v_r": v_r_e, "v_t": v_t_e, "eta": I_e, "a": a_e,
           "b": b_e, "d": d_e, "tau_s": tau_ampa, "v": v_t_e, "Delta": Delta_e, "E_e": E_ampa, "E_i": E_gaba}
in_vars = {"C": C_i, "k": k_i, "v_r": v_r_i, "v_t": v_t_i, "eta": I_i, "a": a_i,
           "b": b_i, "d": d_i, "tau_s": tau_gaba, "v": v_t_i, "Delta": Delta_i, "E_e": E_ampa, "E_i": E_gaba}
tc_vars = {"C": C_t, "k": k_t, "v_r": v_r_t, "v_t": v_t_t, "eta": I_t, "a": a_t,
           "b": b_t, "d": d_t, "tau_s": tau_ampa, "v": v_t_t, "Delta": Delta_t, "E_e": E_ampa, "E_i": E_gaba}
neuron_params = {"rs": pc_vars, "lts": in_vars, "tc": tc_vars}
for node in node_names:
    net.update_var(node_vars={f"{node}/ik_op/{var}": val for var, val in neuron_params[node].items()})

# model dynamics
################

# define model input
T = 3000.0
dt = 1e-3
dts = 1.0
start = 1000.0
stop = 2000.0
inp = np.zeros((int(T/dt),))
inp[int(start/dt):int(stop/dt)] = 50.0

# perform simulation
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=dts, solver="scipy", method="RK45", atol=1e-6,
              rtol=1e-6, inputs={"tc/ik_op/I_ext": inp}, outputs={key: f"{key}/ik_op/s" for key in node_names},
              clear=True)

# plotting
fig, ax = plt.subplots(figsize=(12, 4))
res.plot(ax=ax)
ax.set_xlabel("time (ms)")
ax.set_ylabel("s")
plt.tight_layout()
plt.show()
