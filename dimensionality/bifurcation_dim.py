from pycobi import ODESystem
from pyrates import CircuitTemplate
import sys
import matplotlib.pyplot as plt

"""
Bifurcation analysis of the Izhikevich mean-field model.

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) and 
auto-07p (https://github.com/auto-07p/auto-07p) installed.
"""

# preparations
##############

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
eta = 50.0
Delta = 1.0
g = 15.0
a = 0.03
b = -2.0
d = 40.0
E_r = 0.0
tau_s = 6.0
node_vars = {"C": C, "k": k, "v_r": v_r, "v_t": v_t, "eta": eta, "a": a, "b": b, "d": d, "E_r": E_r, "tau_s": tau_s,
             "Delta": Delta, "g": g}

# create circuit
net = CircuitTemplate.from_yaml("config/ik_mf/ik")
net.update_var(node_vars={f"p/ik_op/{key}": val for key, val in node_vars.items()})

# config
n_dim = 4
n_params = 20
ode = ODESystem.from_template(net, working_dir="../Izhikevich/config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = ode.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                         EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# continuation in g
c1_sols, c1_cont = ode.run(starting_point='UZ1', c='qif', ICP="p/ik_op/g", NPAR=n_params, NDIM=n_dim, name='g:1',
                           origin=t_cont, NMX=8000, DSMAX=0.05, UZR={"p/ik_op/g": [5.0, 25.0]}, STOP=[], NPR=100,
                           RL1=100.0, RL0=0.0, bidirectional=True)

# continuations in Delta
c2_sols, c2_cont = ode.run(starting_point='UZ1', c='qif', ICP="p/ik_op/Delta", NPAR=n_params, NDIM=n_dim, name='g:1',
                           origin=c1_cont, NMX=8000, DSMAX=0.05, UZR={}, STOP=[], NPR=10,
                           RL1=10.0, RL0=0.0, bidirectional=True)
c3_sols, c3_cont = ode.run(starting_point='UZ2', c='qif', ICP="p/ik_op/Delta", NPAR=n_params, NDIM=n_dim, name='g:1',
                           origin=c1_cont, NMX=8000, DSMAX=0.05, UZR={}, STOP=[], NPR=10,
                           RL1=10.0, RL0=0.0, bidirectional=True)

# 2D continuation in g and Delta
# ode.run(starting_point='LP1', c='qif2', ICP=[5, 8], name='D/I:lp1', origin=r1_cont, NMX=8000, DSMAX=0.05,
#       NPR=20, RL1=5.0, RL0=0.0, bidirectional=True)
# ode.run(starting_point='LP2', c='qif2', ICP=[5, 8], name='D/I:lp2', origin=r1_cont, NMX=8000, DSMAX=0.05,
#       NPR=20, RL1=5.0, RL0=0.0, bidirectional=True)

# plot results

