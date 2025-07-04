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
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
Delta = 2.0
eta = 0.0
b = -10.0
kappa = 0.0
alpha = 0.0
gamma = 200.0
theta = 0.02
mu = 0.5
tau_a = 1000.0
tau_u = 100.0
g = 0.0
E_r = 0.0

params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'alpha': alpha,
    'tau_a': tau_a, 'tau_u': tau_u, 'g': g, 'E_r': E_r, 'b': b, 's': gamma, 'theta': theta, 'mu': mu
}

# initialize model
op = "ik_ca_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/ik_ca")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in params.items()})

# config
n_dim = 5
n_params = 22
# ode = ODESystem.from_template(ik, working_dir="../organoid_dynamics/config", auto_dir=auto_dir, init_cont=False)
ode = ODESystem(eq_file="ik_ca_equations", working_dir="../organoid_dynamics/config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = ode.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                         EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 4500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# choose coupling strength
ode.run(starting_point='UZ1', c='1d', ICP=22, NPAR=n_params, NDIM=n_dim, name="g:0",
        origin=t_cont, NMX=4000, DSMAX=0.05, UZR={22: [40.0]}, STOP=["UZ1"],
        NPR=100, RL1=100.0, RL0=0.0, bidirectional=False, EPSS=1e-4)

# choose SFA strength
ode.run(starting_point='UZ1', c='1d', ICP=16, NPAR=n_params, NDIM=n_dim, name="kappa:0",
        origin="g:0", NMX=4000, DSMAX=0.05, UZR={16: [0.5]}, STOP=["UZ1"],
        NPR=100, RL1=10.0, RL0=0.0, bidirectional=False, EPSS=1e-4)

# choose CA threshold
ode.run(starting_point='UZ1', c='1d', ICP=18, NPAR=n_params, NDIM=n_dim, name="theta:0",
        origin="kappa:0", NMX=4000, DSMAX=0.05, UZR={18: [10.0]}, STOP=["UZ1"],
        NPR=100, RL1=100.0, RL0=0.0, EPSS=1e-4)

# continuation in independent parameter
p1 = "alpha"
p1_idx = 21
p1_vals = [0.1, 0.2, 0.4]
c1_sols, c1_cont = ode.run(starting_point='UZ1', c='1d', ICP=p1_idx, NPAR=n_params, NDIM=n_dim, name='mu:0',
                           origin="theta:0", NMX=8000, DSMAX=0.2, UZR={p1_idx: p1_vals}, STOP=[],
                           NPR=100, RL1=100.0, RL0=0.0, EPSS=1e-4)

# continuations in eta
for i, p1_val in enumerate(p1_vals):

    c2_sols, c2_cont = ode.run(starting_point=f'UZ{i+1}', ICP=8, name=f'eta:{i+1}', DSMAX=0.1,
                               origin=c1_cont, UZR={}, STOP=[], NPR=10, RL1=150.0, RL0=-50.0, bidirectional=True)

    try:
        ode.run(starting_point="HB1", ICP=[8, 11], name=f"eta:{i+1}:lc:1", origin=c2_cont, ISW=-1, IPS=2, NMX=8000,
                DSMAX=0.1, NCOL=6, NTST=200, STOP=["LP4"], EPSL=1e-8, EPSU=1e-8, EPSS=1e-5)
    except KeyError:
        pass
    try:
        ode.run(starting_point="HB2", ICP=[8, 11], name=f"eta:{i+1}:lc:2", origin=c2_cont, ISW=-1, IPS=2, NMX=8000,
                DSMAX=0.1, NCOL=6, NTST=200, STOP=["LP4"], EPSL=1e-8, EPSU=1e-8, EPSS=1e-5)
    except KeyError:
        pass

    # plot 1D bifurcation diagram
    fig, ax = plt.subplots(figsize=(12, 4))
    ode.plot_continuation("PAR(8)", "U(1)", cont=f"eta:{i+1}", ax=ax, bifurcation_legend=True)
    try:
        ode.plot_continuation("PAR(8)", "U(1)", cont=f"eta:{i+1}:lc:1", ax=ax, bifurcation_legend=False)
    except (KeyError, AxisError):
        pass
    try:
        ode.plot_continuation("PAR(8)", "U(1)", cont=f"eta:{i+1}:lc:2", ax=ax, bifurcation_legend=False)
    except (KeyError, AxisError):
        pass
    ax.set_title(f"1D bifurcation diagram for {p1} = {p1_val}")
    plt.tight_layout()

# 2D continuation I
p1_val_idx = 2
# ode.run(starting_point='LP1', ICP=[p1_idx, 8], name=f'{p1}/eta:lp1', origin="eta:2", NMX=4000, DSMAX=0.2,
#         NPR=10, RL1=100.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
# ode.run(starting_point='LP2', ICP=[16, 9], name=f'{p1}/eta:lp2', origin="eta:2", NMX=4000, DSMAX=0.2,
#         NPR=10, RL1=300.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
ode.run(starting_point='HB1', ICP=[p1_idx, 8], name=f'{p1}/eta:hb1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.2,
        NPR=50, RL1=100.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
# ode.run(starting_point='HB2', ICP=[p1_idx, 8], name=f'{p1}/eta:hb2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.2,
#         NPR=50, RL1=100.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])

# 2D continuation II
p2 = "mu"
p2_idx = 20
# ode.run(starting_point='LP1', ICP=[p2_idx, 8], name=f'{p2}/eta:lp1', origin="eta:2", NMX=4000, DSMAX=0.2,
#         NPR=10, RL1=1.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
ode.run(starting_point='HB1', ICP=[p2_idx, 8], name=f'{p2}/eta:hb1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.2,
        NPR=50, RL1=1.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
# ode.run(starting_point='HB2', ICP=[p2_idx, 8], name=f'{p2}/eta:hb2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.2,
#         NPR=50, RL1=1.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])

# 2D continuation II
p3 = "kappa"
p3_idx = 16
# ode.run(starting_point='LP1', ICP=[p2_idx, 8], name=f'{p2}/eta:lp1', origin="eta:2", NMX=4000, DSMAX=0.2,
#         NPR=10, RL1=1.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
ode.run(starting_point='HB1', ICP=[p3_idx, 8], name=f'{p3}/eta:hb1', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.2,
        NPR=50, RL1=10.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])
# ode.run(starting_point='HB2', ICP=[p2_idx, 8], name=f'{p2}/eta:hb2', origin=f"eta:{p1_val_idx+1}", NMX=4000, DSMAX=0.2,
#         NPR=50, RL1=1.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400, STOP=["BP2"])

# plot 2D bifurcation diagrams
fig, ax = plt.subplots(figsize=(12, 4))
# ode.plot_continuation("PAR(8)", f"PAR({p1_idx})", cont=f"{p1}/eta:lp1", ax=ax, bifurcation_legend=True)
ode.plot_continuation("PAR(8)", f"PAR({p1_idx})", cont=f"{p1}/eta:hb1", ax=ax, bifurcation_legend=True,
                      line_color_stable="green")
fig.suptitle(f"2d bifurcations: {p1}/eta for {p1} = {p1_vals[p1_val_idx]}")
plt.tight_layout()
fig, ax = plt.subplots(figsize=(12, 4))
# ode.plot_continuation("PAR(8)", f"PAR({p2_idx})", cont=f"{p2}/eta:lp1", ax=ax, bifurcation_legend=True)
ode.plot_continuation("PAR(8)", f"PAR({p2_idx})", cont=f"{p2}/eta:hb1", ax=ax, bifurcation_legend=True,
                      line_color_stable="green")
fig.suptitle(f"2d bifurcations: {p2}/eta for {p1} = {p1_vals[p1_val_idx]}")
plt.tight_layout()

plt.show()
