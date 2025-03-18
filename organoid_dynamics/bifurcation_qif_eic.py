from numpy.exceptions import AxisError
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

# exc parameters
exc_params = {
    'tau': 10.0, 'Delta': 0.5, 'eta': 0.0, 'kappa': 0.0, 'tau_u': 500.0,
    'J_e': 15.0, 'J_i': 0.0, 'tau_s': 6.0
}

# inh parameters
inh_params = {
    'tau': 20.0, 'Delta': 1.0, 'eta': 0.0, 'J_e': 10.0, 'J_i': 10.0, 'tau_s': 20.0
}

# initialize model
exc_op = "qif_sfa_op"
inh_op = "qif_sfa_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/eic_ik")

# update parameters
ik.update_var(node_vars={f"exc/{exc_op}/{var}": val for var, val in exc_params.items()})
ik.update_var(node_vars={f"inh/{inh_op}/{var}": val for var, val in inh_params.items()})

# config
n_dim = 8
n_params = 20
ode = ODESystem.from_template(ik, working_dir="../organoid_dynamics/config", auto_dir=auto_dir, init_cont=False)
# ode = ODESystem(eq_file="qif_equations", working_dir="../organoid_dynamics/config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = ode.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                         EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 4500.0}, STOP={'UZ1'})

# identify parameter of interest
eta = 16
kappa = 18
tau_u = 17
g_e = 5

########################
# bifurcation analysis #
########################

# continuation in kappa
kappas = [600.0, 900.0, 1200.0]
c1_sols, c1_cont = ode.run(starting_point='UZ1', c='1d', ICP=kappa, NPAR=n_params, NDIM=n_dim, name='kappa',
                           origin=t_cont, NMX=5000, DSMAX=0.5, UZR={kappa: kappas}, STOP=[],
                           NPR=100, RL1=2100.0, RL0=0.0, bidirectional=True, EPSS=1e-4)

# continuations in eta
for i, k in enumerate(kappas):

    c2_sols, c2_cont = ode.run(starting_point=f'UZ{i+1}', ICP=eta, name=f'eta:{i+1}', DSMAX=0.1,
                               origin=c1_cont, UZR={}, STOP=[], NPR=10, RL1=250.0, RL0=0.0, bidirectional=True)
    for j in range(2):
        try:
            ode.run(starting_point=f"HB{j+1}", ICP=[eta, 11], name=f"eta:{i+1}:lc:{j+1}", origin=c2_cont, ISW=-1, IPS=2,
                    NMX=8000, DSMAX=0.5, NCOL=4, NTST=100, STOP=["LP4"], EPSL=1e-6, EPSU=1e-6, EPSS=1e-4)
        except KeyError:
            pass

    # plot 1D bifurcation diagram
    fig, ax = plt.subplots(figsize=(12, 4))
    ode.plot_continuation(f"PAR({eta})", "U(1)", cont=f"eta:{i+1}", ax=ax, bifurcation_legend=True)
    for j in range(2):
        try:
            ode.plot_continuation(f"PAR({eta})", "U(1)", cont=f"eta:{i+1}:lc:{j+1}", ax=ax,
                                  bifurcation_legend=False, ignore=["BP"])
        except (KeyError, AxisError):
            pass
    ax.set_title(f"1D bifurcation diagram for kappa = {k}")
    plt.tight_layout()

# 2D continuation I
p1 = "kappa"
ode.run(starting_point='LP1', ICP=[kappa, eta], name=f'{p1}/eta:lp1', origin="eta:3", NMX=4000, DSMAX=0.5,
        NPR=20, RL1=5000.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, get_stability=False)
ode.run(starting_point='LP2', ICP=[kappa, eta], name=f'{p1}/eta:lp2', origin="eta:3", NMX=4000, DSMAX=0.5,
        NPR=20, RL1=5000.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, get_stability=False)
ode.run(starting_point='HB1', ICP=[kappa, eta], name=f'{p1}/eta:hb1', origin="eta:3", NMX=4000, DSMAX=0.5,
        NPR=20, RL1=5000.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, get_stability=False)
ode.run(starting_point='HB2', ICP=[kappa, eta], name=f'{p1}/eta:hb2', origin="eta:3", NMX=4000, DSMAX=0.5,
        NPR=20, RL1=5000.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, get_stability=False)

# 2D continuation II
p2 = "tau_u"
ode.run(starting_point='LP1', ICP=[tau_u, eta], name=f'{p2}/eta:lp1', origin="eta:3", NMX=4000, DSMAX=0.5,
        NPR=20, RL1=5000.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, get_stability=False)
ode.run(starting_point='LP2', ICP=[tau_u, eta], name=f'{p2}/eta:lp2', origin="eta:3", NMX=4000, DSMAX=0.5,
        NPR=20, RL1=5000.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, get_stability=False)
ode.run(starting_point='HB1', ICP=[tau_u, eta], name=f'{p2}/eta:hb1', origin="eta:3", NMX=4000, DSMAX=0.5,
        NPR=20, RL1=5000.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, get_stability=False)
ode.run(starting_point='HB2', ICP=[tau_u, eta], name=f'{p2}/eta:hb2', origin="eta:3", NMX=4000, DSMAX=0.5,
        NPR=20, RL1=5000.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, get_stability=False)

# 2D continuation III
p3 = "g_e"
ode.run(starting_point='LP1', ICP=[g_e, eta], name=f'{p3}/eta:lp1', origin="eta:3", NMX=4000, DSMAX=0.5,
        NPR=20, RL1=500.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, get_stability=False)
ode.run(starting_point='LP2', ICP=[g_e, eta], name=f'{p3}/eta:lp2', origin="eta:3", NMX=4000, DSMAX=0.5,
        NPR=20, RL1=500.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, get_stability=False)
ode.run(starting_point='HB1', ICP=[g_e, eta], name=f'{p3}/eta:hb1', origin="eta:3", NMX=4000, DSMAX=0.5,
        NPR=20, RL1=500.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, get_stability=False)
ode.run(starting_point='HB2', ICP=[g_e, eta], name=f'{p3}/eta:hb2', origin="eta:3", NMX=4000, DSMAX=0.5,
        NPR=20, RL1=500.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, get_stability=False)

# plot 2D bifurcation diagrams
for param, idx in zip([p1, p2, p3], [kappa, tau_u, g_e]):
    fig, ax = plt.subplots(figsize=(12, 4))
    ode.plot_continuation(f"PAR({eta})", f"PAR({idx})", cont=f"{param}/eta:lp1", ax=ax, bifurcation_legend=True,
                          get_stability=False)
    ode.plot_continuation(f"PAR({eta})", f"PAR({idx})", cont=f"{param}/eta:lp2", ax=ax, bifurcation_legend=True,
                          get_stability=False)
    for j in range(2):
        try:
            ode.plot_continuation(f"PAR({eta})", f"PAR({idx})", cont=f"{param}/eta:hb{j+1}", ax=ax,
                                  bifurcation_legend=True, line_color_stable="green", get_stability=False)
        except KeyError:
            pass
    fig.suptitle(f"2d bifurcations: {param}/eta")
    plt.tight_layout()
plt.show()
