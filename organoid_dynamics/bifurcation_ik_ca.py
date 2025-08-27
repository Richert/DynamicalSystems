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

# plotting params
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 16.0
plt.rcParams['axes.titlesize'] = 16.0
plt.rcParams['axes.labelsize'] = 16.0
plt.rcParams['lines.linewidth'] = 2.0
markersize = 8

# directories
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# model parameters
node_vars = {"C": 100.0, "k": 0.7, "v_r": -60.0, "v_t": -40.0, "eta": -4.0, "Delta": 1.0, "g": 20.0,
             "alpha": 0.1, "tau_a": 100.0, "A0": 0.5, "kappa": 0.0, "tau_u": 50.0, "b": -2.0, "tau_x": 500.0}

# initialize model
op = "ik_ca_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/ik_ca")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in node_vars.items()})

# config
n_dim = 5
n_params = 21
ode = ODESystem.from_template(ik, working_dir="config", auto_dir=auto_dir,
                              init_cont=False)
# ode = ODESystem(eq_file="ik_ca_equations", working_dir="../organoid_dynamics/config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = ode.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                         EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 4500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# continuation in independent parameter
p1 = "kappa"
p1_idx = 17
p1_vals = [5.0, 10.0, 20.0]
p1_min, p1_max = 0.0, 200.0
c1_sols, c1_cont = ode.run(starting_point='UZ1', c='1d', ICP=p1_idx, NPAR=n_params, NDIM=n_dim, name=f'{p1}:0',
                           origin="t", NMX=8000, DSMAX=0.1, UZR={p1_idx: p1_vals}, STOP=[],
                           NPR=20, RL1=p1_max, RL0=p1_min, EPSS=1e-4, bidirectional=True)

# continuations in eta
eta_idx = 9
eta_min, eta_max = -10.0, 100.0
for i, p1_val in enumerate(p1_vals):

    c2_sols, c2_cont = ode.run(starting_point=f'UZ{i+1}', ICP=eta_idx, name=f'eta:{i+1}', DSMAX=0.1,
                               origin=c1_cont, UZR={}, STOP=[], NPR=5, RL1=eta_max, RL0=eta_min, bidirectional=True,
                               variables=["U(1)"])

    try:
        ode.run(starting_point="HB1", ICP=[eta_idx, 11], name=f"eta:{i+1}:lc:1", origin=c2_cont, ISW=-1, IPS=2, NMX=6000,
                DSMAX=0.1, NCOL=6, NTST=200, STOP=["LP6", "BP2"], EPSL=1e-7, EPSU=1e-7, EPSS=1e-4, NPR=10,
                variables=["U(1)"])
    except KeyError:
        pass
    try:
        ode.run(starting_point="HB2", ICP=[eta_idx, 11], name=f"eta:{i+1}:lc:2", origin=c2_cont, ISW=-1, IPS=2, NMX=6000,
                DSMAX=0.1, NCOL=6, NTST=200, STOP=["LP6", "BP2"], EPSL=1e-7, EPSU=1e-7, EPSS=1e-4, NPR=10,
                variables=["U(1)"])
    except KeyError:
        pass

    # plot 1D bifurcation diagram
    fig, ax = plt.subplots(figsize=(12, 4))
    ode.plot_continuation(f"PAR({eta_idx})", "U(1)", cont=f"eta:{i+1}", ax=ax, bifurcation_legend=True,
                          default_size=markersize)
    try:
        ax.get_legend().remove()
        ode.plot_continuation(f"PAR({eta_idx})", "U(1)", cont=f"eta:{i+1}:lc:1", ax=ax, bifurcation_legend=True,
                              ignore=["BP"], default_size=markersize)
    except (KeyError, AxisError):
        pass
    try:
        ax.get_legend().remove()
        ode.plot_continuation(f"PAR({eta_idx})", "U(1)", cont=f"eta:{i+1}:lc:2", ax=ax, bifurcation_legend=True,
                              ignore=["BP"], default_size=markersize)
    except (KeyError, AxisError):
        pass
    ax.set_title(f"1D bifurcation diagram for {p1} = {p1_val}")
    plt.tight_layout()

# 2D continuation I
p1_val_idx = 1
p1_max_step = 0.05
fold_bifurcations = True
hopf_bifurcation_1 = True
hopf_bifurcation_2 = True
pd_bifurcation = True
tr_bifurcation = True
try:
    ode.run(starting_point='LP1', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:lp1', origin=f"eta:{p1_val_idx+1}", NMX=4000,
            DSMAX=p1_max_step, NPR=10, RL1=p1_max, RL0=p1_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400,
            variables=["U(1)"], get_stability=False)
    ode.run(starting_point='LP2', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:lp2', origin=f"eta:{p1_val_idx+1}",
            NMX=4000, DSMAX=p1_max_step, NPR=10, RL1=p1_max, RL0=p1_min, bidirectional=True, ILP=0, IPS=1, ISW=2,
            NTST=400, variables=["U(1)"], get_stability=False)
except KeyError:
    fold_bifurcations = False
try:
    ode.run(starting_point='HB1', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:hb1', origin=f"eta:{p1_val_idx+1}", NMX=4000,
            DSMAX=p1_max_step, NPR=10, RL1=p1_max, RL0=p1_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400,
            variables=["U(1)"], get_stability=False)
except KeyError:
    hopf_bifurcation_1 = False
try:
    ode.run(starting_point='HB2', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:hb2', origin=f"eta:{p1_val_idx+1}",
            NMX=4000, DSMAX=p1_max_step, NPR=10, RL1=p1_max, RL0=p1_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400,
            variables=["U(1)"], get_stability=False)
except KeyError:
    hopf_bifurcation_2 = False
try:
    ode.run(starting_point='PD1', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:pd1', origin=f"eta:{p1_val_idx + 1}:lc:2",
            NMX=4000, DSMAX=p1_max_step, NPR=10, RL1=p1_max, RL0=p1_min, bidirectional=True, ILP=0, IPS=2, ISW=2, NTST=400,
            get_stability=False, variables=["U(1)"])
except KeyError:
    pd_bifurcation = False
try:
    ode.run(starting_point='TR1', ICP=[p1_idx, eta_idx], name=f'{p1}/eta:tr1', origin=f"eta:{p1_val_idx + 1}:lc:2",
            NMX=4000, DSMAX=p1_max_step, NPR=10, RL1=p1_max, RL0=p1_min, bidirectional=True, ILP=0, IPS=2, ISW=2, NTST=400,
            get_stability=False, variables=["U(1)"])
except KeyError:
    tr_bifurcation = False

# 2D continuation II
p2 = "alpha"
p2_idx = 19
p2_max_step = 0.05
p2_min, p2_max = 0.0, 1.0
if fold_bifurcations:
    ode.run(starting_point='LP1', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:lp1', origin=f"eta:{p1_val_idx+1}",
            NMX=4000, DSMAX=p2_max_step, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400,
            get_stability=False, STOP=["BP2"], variables=["U(1)"])
    ode.run(starting_point='LP2', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:lp2', origin=f"eta:{p1_val_idx+1}", NMX=4000,
            DSMAX=p2_max_step, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"],
            variables=["U(1)"], get_stability=False)
if hopf_bifurcation_1:
    ode.run(starting_point='HB1', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:hb1', origin=f"eta:{p1_val_idx+1}", NMX=4000,
            DSMAX=p2_max_step, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"],
            variables=["U(1)"], get_stability=False)
if hopf_bifurcation_2:
    ode.run(starting_point='HB2', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:hb2', origin=f"eta:{p1_val_idx+1}", NMX=4000,
            DSMAX=p2_max_step, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"],
            variables=["U(1)"], get_stability=False)
if pd_bifurcation:
    ode.run(starting_point='PD1', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:pd1', origin=f"eta:{p1_val_idx+1}:lc:2",
            NMX=4000, DSMAX=p2_max_step, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=2, ISW=2, NTST=400,
            get_stability=False, variables=["U(1)"])
if tr_bifurcation:
    ode.run(starting_point='TR1', ICP=[p2_idx, eta_idx], name=f'{p2}/eta:tr1', origin=f"eta:{p1_val_idx+1}:lc:2",
            NMX=4000, DSMAX=p2_max_step, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=2, ISW=2, NTST=400,
            get_stability=False, variables=["U(1)"])

# 2D continuation III
if fold_bifurcations:
    ode.run(starting_point='LP1', ICP=[p2_idx, p1_idx], name=f'{p2}/{p1}:lp1', origin=f"eta:{p1_val_idx+1}", NMX=4000,
            DSMAX=p2_max_step, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400,
            variables=["U(1)"], get_stability=False)
    ode.run(starting_point='LP2', ICP=[p2_idx, p1_idx], name=f'{p2}/{p1}:lp2', origin=f"eta:{p1_val_idx+1}", NMX=4000,
            DSMAX=p2_max_step, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400,
            variables=["U(1)"], get_stability=False)
if hopf_bifurcation_1:
    ode.run(starting_point='HB1', ICP=[p2_idx, p1_idx], name=f'{p2}/{p1}:hb1', origin=f"eta:{p1_val_idx+1}", NMX=4000,
            DSMAX=p2_max_step, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400,
            variables=["U(1)"], get_stability=False)
if hopf_bifurcation_2:
    ode.run(starting_point='HB2', ICP=[p2_idx, p1_idx], name=f'{p2}/{p1}:hb2', origin=f"eta:{p1_val_idx+1}", NMX=4000,
            DSMAX=p2_max_step, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400,
            variables=["U(1)"], get_stability=False)
if pd_bifurcation:
    ode.run(starting_point='PD1', ICP=[p2_idx, p1_idx], name=f'{p2}/{p1}:pd1', origin=f"eta:{p1_val_idx+1}:lc:2",
            NMX=4000, DSMAX=p2_max_step, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=2, ISW=2, NTST=400,
            get_stability=False, variables=["U(1)"])
if tr_bifurcation:
    ode.run(starting_point='TR1', ICP=[p2_idx, p1_idx], name=f'{p2}/{p1}:tr1', origin=f"eta:{p1_val_idx+1}:lc:2",
            NMX=4000, DSMAX=p2_max_step, NPR=10, RL1=p2_max, RL0=p2_min, bidirectional=True, ILP=0, IPS=2, ISW=2, NTST=400,
            get_stability=False, variables=["U(1)"])

print(f"Fold bifurcations: {'yes' if fold_bifurcations else 'no'}")
print(f"HB1 bifurcation: {'yes' if hopf_bifurcation_1 else 'no'}")
print(f"HB2 bifurcation: {'yes' if hopf_bifurcation_2 else 'no'}")
print(f"PD1 bifurcation: {'yes' if pd_bifurcation else 'no'}")
print(f"TR1 bifurcation: {'yes' if tr_bifurcation else 'no'}")

# plot 2D bifurcation diagrams
for idx1, key1, idx2, key2 in zip([p1_idx, p2_idx, p2_idx], [p1, p2, p2], [eta_idx, eta_idx, p1_idx], ["eta", "eta", p1]):
    fig, ax = plt.subplots(figsize=(12, 4))
    if fold_bifurcations:
        for lp in [1, 2]:
            try:
                ode.plot_continuation(f"PAR({idx2})", f"PAR({idx1})", cont=f"{key1}/{key2}:lp{lp}",
                                      ax=ax, bifurcation_legend=True, get_stability=False, default_size=markersize)
            except KeyError:
                pass
    for hb, hb_idx in zip([hopf_bifurcation_1, hopf_bifurcation_2], [1, 2]):
        if hb:
            try:
                ode.plot_continuation(f"PAR({idx2})", f"PAR({idx1})", cont=f"{key1}/{key2}:hb{hb_idx}",
                                      ax=ax, bifurcation_legend=True, line_color_stable="green",
                                      get_stability=False, default_size=markersize)
            except KeyError:
                pass
    if pd_bifurcation:
        try:
            ode.plot_continuation(f"PAR({idx2})", f"PAR({idx1})", cont=f"{key1}/{key2}:pd1",
                                  ax=ax, bifurcation_legend=True, line_color_stable="blue", line_style_stable="dashed",
                                  ignore=["BP"], get_stability=False, default_size=markersize,
                                  custom_bf_styles={'R1': {'marker': 's', 'color': 'k'},
                                                    'R2': {'marker': 'o', 'color': 'k'},
                                                    'R3': {'marker': 'v', 'color': 'k'},
                                                    'R4': {'marker': 'd', 'color': 'k'}},
                                  )
        except KeyError:
            pass
    if tr_bifurcation:
        try:
            ode.plot_continuation(f"PAR({idx2})", f"PAR({idx1})", cont=f"{key1}/{key2}:tr1",
                                  ax=ax, bifurcation_legend=True, line_color_stable="red", line_style_stable="dashed",
                                  ignore=["BP"], get_stability=False, default_size=markersize,
                                  custom_bf_styles={'R1': {'marker': 's', 'color': 'k'},
                                                    'R2': {'marker': 'o', 'color': 'k'},
                                                    'R3': {'marker': 'v', 'color': 'k'},
                                                    'R4': {'marker': 'd', 'color': 'k'}},
                                  )
        except KeyError:
            pass
    fig.suptitle(f"2d bifurcations: {key1}/{key2} for {p1} = {p1_vals[p1_val_idx]}")
    plt.tight_layout()

plt.show()
