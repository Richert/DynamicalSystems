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
node_vars = {"tau": 1.0, "J": 20.0, "eta": -2.0, "tau_s": 0.5, "Delta": 0.5, "tau_a": 50.0, "kappa": 0.04}

# initialize model
op = "qif_ca_op"
ik = CircuitTemplate.from_yaml("config/qif_mf/qif_ca")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in node_vars.items()})

# config
n_dim = 4
n_params = 8
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
p1 = "Delta"
p1_idx = 2
p1_vals = [0.2, 0.6, 1.0, 1.4]
c1_sols, c1_cont = ode.run(starting_point='UZ1', c='1d', ICP=p1_idx, NPAR=n_params, NDIM=n_dim, name=f'{p1}:0',
                           origin="t", NMX=8000, DSMAX=0.1, UZR={p1_idx: p1_vals}, STOP=[],
                           NPR=20, RL1=2.0, RL0=0.0, EPSS=1e-4, bidirectional=True)

# continuations in eta
eta_idx = 5
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
ode.run(starting_point='LP1', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:lp1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
        NPR=10, RL1=2.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
ode.run(starting_point='LP2', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:lp2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
        NPR=10, RL1=2.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
ode.run(starting_point='HB1', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:hb1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
        NPR=10, RL1=2.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
ode.run(starting_point='HB2', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:hb2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
        NPR=10, RL1=2.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])

# 2D continuation II
p2 = "kappa"
p2_idx = 8
ode.run(starting_point='LP1', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:lp1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
        NPR=10, RL1=1.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
ode.run(starting_point='LP2', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:lp2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
        NPR=10, RL1=1.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
ode.run(starting_point='HB1', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:hb1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
        NPR=10, RL1=1.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
ode.run(starting_point='HB2', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:hb2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
        NPR=10, RL1=1.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])

# 2D continuation III
p3 = "J"
p3_idx = 3
ode.run(starting_point='LP1', ICP=[p3_idx, eta_idx], name=f'{p3}/eta:lp1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
        NPR=10, RL1=50.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
ode.run(starting_point='LP2', ICP=[p3_idx, eta_idx], name=f'{p3}/eta:lp2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
        NPR=10, RL1=50.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
ode.run(starting_point='HB1', ICP=[p3_idx, eta_idx], name=f'{p3}/eta:hb1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
        NPR=10, RL1=50.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
ode.run(starting_point='HB2', ICP=[p3_idx, eta_idx], name=f'{p3}/eta:hb2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.02,
        NPR=10, RL1=50.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])

# plot 2D bifurcation diagrams
fig, ax = plt.subplots(figsize=(12, 4))
ode.plot_continuation(f"PAR({eta_idx})", f"PAR({p1_idx})", cont=f"{p1}/eta:lp1", ax=ax, bifurcation_legend=True)
ode.plot_continuation(f"PAR({eta_idx})", f"PAR({p1_idx})", cont=f"{p1}/eta:lp2", ax=ax, bifurcation_legend=True)
# ode.plot_continuation(f"PAR({eta_idx})", f"PAR({p1_idx})", cont=f"{p1}/eta:hb1", ax=ax, bifurcation_legend=True,
#                       line_color_stable="green")
ode.plot_continuation(f"PAR({eta_idx})", f"PAR({p1_idx})", cont=f"{p1}/eta:hb2", ax=ax, bifurcation_legend=True,
                      line_color_stable="green")

fig.suptitle(f"2d bifurcations: {p1}/eta for {p1} = {p1_vals[p1_val_idx]}")
plt.tight_layout()
fig, ax = plt.subplots(figsize=(12, 4))
ode.plot_continuation(f"PAR({eta_idx})", f"PAR({p2_idx})", cont=f"{p2}/eta:lp1", ax=ax, bifurcation_legend=True)
ode.plot_continuation(f"PAR({eta_idx})", f"PAR({p2_idx})", cont=f"{p2}/eta:lp2", ax=ax, bifurcation_legend=True)
# ode.plot_continuation(f"PAR({eta_idx})", f"PAR({p2_idx})", cont=f"{p2}/eta:hb1", ax=ax, bifurcation_legend=True,
#                       line_color_stable="green")
ode.plot_continuation(f"PAR({eta_idx})", f"PAR({p2_idx})", cont=f"{p2}/eta:hb2", ax=ax, bifurcation_legend=True,
                      line_color_stable="green")
fig.suptitle(f"2d bifurcations: {p2}/eta for {p1} = {p1_vals[p1_val_idx]}")
plt.tight_layout()
fig, ax = plt.subplots(figsize=(12, 4))
ode.plot_continuation(f"PAR({eta_idx})", f"PAR({p3_idx})", cont=f"{p3}/eta:lp1", ax=ax, bifurcation_legend=True)
ode.plot_continuation(f"PAR({eta_idx})", f"PAR({p3_idx})", cont=f"{p3}/eta:lp2", ax=ax, bifurcation_legend=True)
# ode.plot_continuation(f"PAR({eta_idx})", f"PAR({p3_idx})", cont=f"{p3}/eta:hb1", ax=ax, bifurcation_legend=True,
#                       line_color_stable="green")
ode.plot_continuation(f"PAR({eta_idx})", f"PAR({p3_idx})", cont=f"{p3}/eta:hb2", ax=ax, bifurcation_legend=True,
                      line_color_stable="green")
fig.suptitle(f"2d bifurcations: {p3}/eta for {p1} = {p1_vals[p1_val_idx]}")
plt.tight_layout()
plt.show()
