import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from rectipy import FeedbackNetwork, random_connectivity
from pyrates import CircuitTemplate, NodeTemplate

# load data
###########

# load the training and testing data
path = os.path.expanduser("~/OneDrive/data/SHD")
train_data = h5py.File(os.path.join(path, "shd_train.h5"), "r")
test_data = h5py.File(os.path.join(path, "shd_test.h5"), "r")

# extract spikes and labels from data set
X_train = train_data["spikes"]
y_train = train_data["labels"]
X_test = test_data["spikes"]
y_test = test_data["labels"]

# parameters
############

# general parameters
dt = 1e-3
device = "cpu"

# network dimensions
n_e = 800
n_i = 200
n_t = 50
n_r = 50
n_s = 100
n_in = 100
n_out = 20

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

# RTN neuron parameters
C_r = 40.0   # unit: pF
k_r = 0.25  # unit: None
v_r_r = -65.0  # unit: mV
v_t_r = -45.0  # unit: mV
Delta_r = 0.4  # unit: mV
d_r = 50.0  # unit: pA
a_r = 0.015  # unit: 1/ms
b_r = 10.0  # unit: nS
I_r = 60.0  # unit: pA

# SNR neuron parameters
C_s = 50.0   # unit: pF
k_s = 0.25  # unit: None
v_r_s = -55.0  # unit: mV
v_t_s = -44.0  # unit: mV
Delta_s = 1.0  # unit: mV
d_s = 20.0  # unit: pA
a_s = 1.0  # unit: 1/ms
b_s = 0.25  # unit: nS
I_s = 100.0  # unit: pA

# synaptic parameters
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 5.0
tau_gaba = 8.0

# population-level coupling strengths
J = 50.0
J_e = 1.0*J
J_i = 0.2*J
J_t = 0.2*J
J_r = 0.2*J
J_s = 0.2*J

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
J_ts = 0.4*J_t

# RTN inputs
J_rt = 0.4*J_r
J_rr = 0.2*J_r
J_re = 0.4*J_r

# SNR inputs
J_se = 0.6*J_s
J_st = 0.4*J_s

# create network
################

# initialize network
node = "config/ik_snn/ik"
net = FeedbackNetwork(dt=dt, device=device)

# add network nodes
net.add_diffeq_node("rs", node, input_var="s")