from csv import excel

from pycobi import ODESystem
from pyrates import CircuitTemplate
import sys
import matplotlib.pyplot as plt
from numpy.exceptions import AxisError

"""
Bifurcation analysis of a Calcium-dependent Izhikevich mean-field model.

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) and 
auto-07p (https://github.com/auto-07p/auto-07p) installed.
"""

# preparations
##############

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# model parameters
node_vars = {"tau": 1.0, "J": 20.0, "eta": -2.0, "tau_u": 10.0, "alpha": 0.0, "Delta": 0.5, "tau_a": 100.0, "kappa": 0.0}

# initialize model
op = "qif_ca_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/qif_ca")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in node_vars.items()})

# config
n_dim = 4
n_params = 16
ode = ODESystem.from_template(ik, working_dir="../adaptive_heterogeneous_networks/config", auto_dir=auto_dir,
                              init_cont=False)
# ode = ODESystem(eq_file="ik_ca_equations", working_dir="../organoid_dynamics/config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = ode.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                         EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 4500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# continuation in independent parameter
p1 = "alpha"
p1_idx = 9
p1_vals = [0.0125, 0.025, 0.05]
c1_sols, c1_cont = ode.run(starting_point='UZ1', c='1d', ICP=p1_idx, NPAR=n_params, NDIM=n_dim, name=f'{p1}:0',
                           origin="t", NMX=8000, DSMAX=0.1, UZR={p1_idx: p1_vals}, STOP=[],
                           NPR=20, RL1=2.0, RL0=0.0, EPSS=1e-4, bidirectional=True)

# continuations in eta
eta_idx = 7
for i, p1_val in enumerate(p1_vals):

    c2_sols, c2_cont = ode.run(starting_point=f'UZ{i+1}', ICP=eta_idx, name=f'eta:{i+1}', DSMAX=0.02,
                               origin=c1_cont, UZR={}, STOP=[], NPR=10, RL1=10.0, RL0=-10.0, bidirectional=True)

    try:
        ode.run(starting_point="HB1", ICP=[eta_idx, 11], name=f"eta:{i+1}:lc:1", origin=c2_cont, ISW=-1, IPS=2, NMX=8000,
                DSMAX=0.04, NCOL=6, NTST=200, STOP=["LP4", "BP2"], EPSL=1e-8, EPSU=1e-8, EPSS=1e-5)
    except KeyError:
        pass
    try:
        ode.run(starting_point="HB2", ICP=[eta_idx, 11], name=f"eta:{i+1}:lc:2", origin=c2_cont, ISW=-1, IPS=2, NMX=8000,
                DSMAX=0.04, NCOL=6, NTST=200, STOP=["LP4", "BP2"], EPSL=1e-8, EPSU=1e-8, EPSS=1e-5)
    except KeyError:
        pass

    # plot 1D bifurcation diagram
    fig, ax = plt.subplots(figsize=(12, 4))
    ode.plot_continuation(f"PAR({eta_idx})", "U(1)", cont=f"eta:{i+1}", ax=ax, bifurcation_legend=True)
    try:
        ode.plot_continuation(f"PAR({eta_idx})", "U(1)", cont=f"eta:{i+1}:lc:1", ax=ax, bifurcation_legend=False)
    except (KeyError, AxisError):
        pass
    try:
        ode.plot_continuation(f"PAR({eta_idx})", "U(1)", cont=f"eta:{i+1}:lc:2", ax=ax, bifurcation_legend=False)
    except (KeyError, AxisError):
        pass
    ax.set_title(f"1D bifurcation diagram for {p1} = {p1_val}")
    plt.tight_layout()

# 2D continuation I
p1_val_idx = 2
fold_bifurcations = True
hopf_bifurcation_1 = True
hopf_bifurcation_2 = True
try:
    ode.run(starting_point='LP1', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:lp1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
            NPR=10, RL1=2.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
    ode.run(starting_point='LP2', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:lp2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
            NPR=10, RL1=2.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
except KeyError:
    fold_bifurcations = False
try:
    ode.run(starting_point='HB1', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:hb1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
            NPR=10, RL1=2.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
except KeyError:
    hopf_bifurcation_1 = False
try:
    ode.run(starting_point='HB2', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:hb2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
            NPR=10, RL1=2.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
except KeyError:
    hopf_bifurcation_2 = False

# 2D continuation II
p2 = "kappa"
p2_idx = 16
p2_min, p2_max = -2.0, 2.0
if fold_bifurcations:
    ode.run(starting_point='LP1', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:lp1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.1,
            NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
    ode.run(starting_point='LP2', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:lp2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.1,
            NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
if hopf_bifurcation_1:
    ode.run(starting_point='HB1', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:hb1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.1,
            NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
if hopf_bifurcation_2:
    ode.run(starting_point='HB2', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:hb2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.1,
            NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])

# 2D continuation III
if fold_bifurcations:
    ode.run(starting_point='LP1', ICP=[p2_idx, p1_idx], name=f'{p2}/{p1}:lp1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.1,
            NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
    ode.run(starting_point='LP2', ICP=[p2_idx, p1_idx], name=f'{p2}/{p1}:lp2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.1,
            NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
if hopf_bifurcation_1:
    ode.run(starting_point='HB1', ICP=[p2_idx, p1_idx], name=f'{p2}/{p1}:hb1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.1,
            NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
if hopf_bifurcation_2:
    ode.run(starting_point='HB2', ICP=[p2_idx, p1_idx], name=f'{p2}/{p1}:hb2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.1,
            NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])

print(f"Fold bifurcations: {'yes' if fold_bifurcations else 'no'}")
print(f"HB1 bifurcation: {'yes' if hopf_bifurcation_1 else 'no'}")
print(f"HB2 bifurcations: {'yes' if hopf_bifurcation_2 else 'no'}")

# plot 2D bifurcation diagrams
for idx1, key1, idx2, key2 in zip([p1_idx, p2_idx, p2_idx], [p1, p2, p2], [eta_idx, eta_idx, p1_idx], ["eta", "eta", p1]):
    fig, ax = plt.subplots(figsize=(12, 4))
    if fold_bifurcations:
        for lp in [1, 2]:
            try:
                ode.plot_continuation(f"PAR({idx2})", f"PAR({idx1})", cont=f"{key1}/{key2}:lp{lp}",
                                      ax=ax, bifurcation_legend=True)
            except KeyError:
                pass
    for hb, hb_idx in zip([hopf_bifurcation_1, hopf_bifurcation_2], [1, 2]):
        if hb:
            try:
                ode.plot_continuation(f"PAR({idx2})", f"PAR({idx1})", cont=f"{key1}/{key2}:hb{hb_idx}",
                                      ax=ax, bifurcation_legend=True, line_color_stable="green")
            except KeyError:
                pass
    fig.suptitle(f"2d bifurcations: {key1}/{key2} for {p1} = {p1_vals[p1_val_idx]}")
    plt.tight_layout()

plt.show()
