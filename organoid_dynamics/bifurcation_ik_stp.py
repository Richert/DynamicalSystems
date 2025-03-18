from pycobi import ODESystem
from pyrates import CircuitTemplate
import sys
import matplotlib.pyplot as plt
from numpy.exceptions import AxisError
import numpy as np

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
Delta = 1.0
eta = 0.0
kappa = 2.0
tau_u = 50.0
g = 60.0
E_r = 0.0
tau_s = 6.0

params = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'tau_u': tau_u,
    'g_e': g, 'E_e': E_r, 'tau_s': tau_s
}

# initialize model
op = "ik_stp_op"
ik = CircuitTemplate.from_yaml("config/ik_mf/ik_stp")

# update parameters
ik.update_var(node_vars={f"p/{op}/{var}": val for var, val in params.items()})

# config
n_dim = 6
n_params = 22
# ode = ODESystem.from_template(ik, working_dir="../organoid_dynamics/config", auto_dir=auto_dir, init_cont=False)
ode = ODESystem(eq_file="stp_equations", working_dir="../organoid_dynamics/config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = ode.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                         EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 4000.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# parameter indices
eta = 17
kappa = 18
g = 4
alpha = 20

# continuation in alpha
alphas = [50, 100.0, 200.0]
c1_sols, c1_cont = ode.run(starting_point='UZ1', c='1d', ICP=alpha, NPAR=n_params, NDIM=n_dim, name='kappa',
                           origin=t_cont, NMX=4000, DSMAX=0.1, UZR={alpha: alphas}, STOP=[],
                           NPR=100, RL1=np.max(alphas) + 10.0, RL0=0.0, bidirectional=True)

# continuations in Delta
for i, a in enumerate(alphas):

    c2_sols, c2_cont = ode.run(starting_point=f'UZ{i+1}', ICP=eta, name=f'eta:{i+1}', DSMAX=0.1, origin=c1_cont,
                               UZR={}, STOP=[], NPR=10, RL1=100.0, RL0=0.0, bidirectional=True)

    for j in range(2):
        try:
            ode.run(starting_point=f"HB{j + 1}", ICP=[eta, 11], name=f"eta:{i+1}:lc:{j+1}", origin=c2_cont, ISW=-1,
                    IPS=2, NMX=8000, DSMAX=0.5, NCOL=4, NTST=100, STOP=["LP4", "BP3"], EPSL=1e-6, EPSU=1e-6, EPSS=1e-4)
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
    ax.set_title(f"1D bifurcation diagram for alpha = {a}")
    plt.tight_layout()

# 2D continuation I
eta_cont = 3
for p, key in zip([kappa, alpha, g], ["kappa", "alpha", "g"]):
    for j in range(2):
        try:
            ode.run(starting_point=f'LP{j+1}', ICP=[p, eta], name=f'{key}/eta:lp{j+1}', origin=f"eta:{eta_cont}",
                    NMX=4000, DSMAX=0.05, NPR=10, RL1=300.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
        except KeyError:
            pass
    for j in range(2):
        try:
            ode.run(starting_point=f'HB{j+1}', ICP=[p, eta], name=f'{key}/eta:hb{j+1}', origin=f"eta:{eta_cont}",
                    NMX=4000, DSMAX=0.05, NPR=10, RL1=300.0, RL0=0.0, bidirectional=True, ILP=0, IPS=1, ISW=2, NTST=400)
        except KeyError:
            pass

# plot 2D bifurcation diagrams
for p, key in zip([kappa, alpha, g], ["kappa", "alpha", "g"]):
    fig, ax = plt.subplots(figsize=(12, 4))
    for j in range(2):
        try:
            ode.plot_continuation(f"PAR({eta})", f"PAR({p})", cont=f"{key}/eta:lp{j+1}", ax=ax,
                                  bifurcation_legend=True)
        except KeyError:
            pass
    for j in range(2):
        try:
            ode.plot_continuation(f"PAR({eta})", f"PAR({p})", cont=f"{key}/eta:hb{j+1}", ax=ax,
                                  bifurcation_legend=True, line_color_stable="green")
        except KeyError:
            pass
    fig.suptitle(f"2d bifurcations: {key}/eta")
    plt.tight_layout()

plt.show()
