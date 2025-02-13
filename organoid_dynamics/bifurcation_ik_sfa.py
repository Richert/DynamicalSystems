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
Delta = 0.5
eta = 0.0
kappa = 2.0
tau_u = 500.0
g = 10.0
E_r = 0.0
tau_s = 6.0

params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'tau_u': tau_u,
    'g': g, 'E_r': E_r, 'tau_s': tau_s
}

# initialize model
op = "ik_sfa_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/ik_sfa")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in params.items()})

# config
n_dim = 5
n_params = 20
# ode = ODESystem.from_template(ik, working_dir="../organoid_dynamics/config", auto_dir=auto_dir, init_cont=False)
ode = ODESystem(eq_file="system_equations", working_dir="../organoid_dynamics/config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = ode.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                         EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 4500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# continuation in kappa
kappas = [0.5, 2.0, 8.0]
c1_sols, c1_cont = ode.run(starting_point='UZ1', c='1d', ICP=16, NPAR=n_params, NDIM=n_dim, name='kappa',
                           origin=t_cont, NMX=4000, DSMAX=0.05, UZR={16: kappas}, STOP=[],
                           NPR=100, RL1=10.0, RL0=0.0, bidirectional=True)

# continuations in Delta
for i, kappa in enumerate(kappas):

    c2_sols, c2_cont = ode.run(starting_point=f'UZ{i+1}', ICP=9, name=f'eta:{i+1}', DSMAX=0.05,
                               origin=c1_cont, UZR={}, STOP=[], NPR=10, RL1=150.0, RL0=0.0, bidirectional=True)

    try:
        ode.run(starting_point="HB1", ICP=[9, 11], name=f"eta:{i+1}:lc:1", origin=c2_cont, ISW=-1, IPS=2, NMX=4000,
                DSMAX=0.1, NCOL=7, NTST=400, STOP=["LP3"])
    except KeyError:
        pass
    try:
        ode.run(starting_point="HB2", ICP=[9, 11], name=f"eta:{i+1}:lc:2", origin=c2_cont, ISW=-1, IPS=2, NMX=4000,
                DSMAX=0.1, NCOL=7, NTST=400, STOP=["LP3"])
    except KeyError:
        pass

    # plot 1D bifurcation diagram
    fig, ax = plt.subplots(figsize=(12, 4))
    ode.plot_continuation("PAR(9)", "U(1)", cont=f"eta:{i+1}", ax=ax, bifurcation_legend=True)
    try:
        ode.plot_continuation("PAR(9)", "U(1)", cont=f"eta:{i+1}:lc:1", ax=ax, bifurcation_legend=False)
    except KeyError:
        pass
    try:
        ode.plot_continuation("PAR(9)", "U(1)", cont=f"eta:{i+1}:lc:2", ax=ax, bifurcation_legend=False)
    except KeyError:
        pass
    ax.set_title(f"1D bifurcation diagram for kappa = {kappa}")
    plt.tight_layout()

# 2D continuation I
p1 = "kappa"
ode.run(starting_point='LP1', ICP=[16, 9], name=f'{p1}/eta:lp1', origin="eta:2", NMX=4000, DSMAX=0.2,
        NPR=10, RL1=20.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
# ode.run(starting_point='LP2', ICP=[16, 9], name=f'{p1}/eta:lp2', origin="eta:2", NMX=4000, DSMAX=0.2,
#         NPR=10, RL1=300.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
ode.run(starting_point='HB1', ICP=[16, 9], name=f'{p1}/eta:hb1', origin="eta:2", NMX=4000, DSMAX=0.2,
        NPR=10, RL1=20.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
ode.run(starting_point='HB2', ICP=[16, 9], name=f'{p1}/eta:hb2', origin="eta:2", NMX=4000, DSMAX=0.2,
        NPR=10, RL1=20.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)

# 2D continuation II
p2 = "tau_u"
ode.run(starting_point='LP1', ICP=[15, 9], name=f'{p2}/eta:lp1', origin="eta:2", NMX=4000, DSMAX=0.2,
        NPR=10, RL1=800.0, RL0=50.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
ode.run(starting_point='HB1', ICP=[15, 9], name=f'{p2}/eta:hb1', origin="eta:2", NMX=4000, DSMAX=0.2,
        NPR=10, RL1=800.0, RL0=50.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
ode.run(starting_point='HB2', ICP=[15, 9], name=f'{p2}/eta:hb2', origin="eta:2", NMX=4000, DSMAX=0.2,
        NPR=10, RL1=800.0, RL0=50.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)

# plot 2D bifurcation diagrams
fig, ax = plt.subplots(figsize=(12, 4))
ode.plot_continuation("PAR(9)", "PAR(16)", cont=f"{p1}/eta:lp1", ax=ax, bifurcation_legend=True)
ode.plot_continuation("PAR(9)", "PAR(16)", cont=f"{p1}/eta:hb1", ax=ax, bifurcation_legend=True,
                      line_color_stable="green")
ode.plot_continuation("PAR(9)", "PAR(16)", cont=f"{p1}/eta:hb2", ax=ax, bifurcation_legend=True,
                      line_color_stable="green")
fig.suptitle(f"2d bifurcations: {p1}/eta for tau_u = 500.0 ms")
plt.tight_layout()
fig, ax = plt.subplots(figsize=(12, 4))
ode.plot_continuation("PAR(9)", "PAR(15)", cont=f"{p2}/eta:lp1", ax=ax, bifurcation_legend=True)
ode.plot_continuation("PAR(9)", "PAR(15)", cont=f"{p2}/eta:hb1", ax=ax, bifurcation_legend=True,
                      line_color_stable="green")
ode.plot_continuation("PAR(9)", "PAR(15)", cont=f"{p2}/eta:hb2", ax=ax, bifurcation_legend=True,
                      line_color_stable="green")
fig.suptitle(f"2d bifurcations: {p2}/eta for kappa = 2.0 pA")
plt.tight_layout()

plt.show()
