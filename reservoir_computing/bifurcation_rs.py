from pycobi import ODESystem
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

"""
Bifurcation analysis of the Izhikevich mean-field model.

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) and 
auto-07p (https://github.com/auto-07p/auto-07p) installed.
"""

# preparations
##############

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 4
n_params = 17
a = ODESystem("config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
a.run(e='rs', c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
      EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# 1D continuations
##################

NPR = 20

# continuation in global coupling strength
a.run(starting_point='UZ1', c='qif', ICP=4, NPAR=n_params, NDIM=n_dim, name='g:1',
      origin='t', NMX=8000, DSMAX=0.1, UZR={4: [15.0]}, STOP=[], NPR=NPR, RL1=50.0, RL0=0.0)

# continuation in Delta
a.run(starting_point='UZ1', c='qif', ICP=5, NPAR=n_params, NDIM=n_dim, name='Delta:1',
      origin='g:1', NMX=8000, DSMAX=0.05, UZR={5: [0.5]}, STOP=[], NPR=NPR, RL1=2.0, RL0=0.0,
      bidirectional=True)

# continuation in SFA strength
a.run(starting_point='UZ1', c='qif', ICP=16, NPAR=n_params, NDIM=n_dim, name='d:1',
      origin='Delta:1', NMX=8000, DSMAX=0.1, UZR={16: [100.0]}, STOP=[], NPR=NPR, RL1=210.0, RL0=0.0,
      bidirectional=True)

# continuation in background input
a.run(starting_point='UZ1', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='eta:1',
      origin='Delta:1', NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=NPR, RL1=200.0, RL0=-200.0)
a.run(starting_point='UZ1', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='eta:2',
      origin='d:1', NMX=8000, DSMAX=0.1, UZR={}, STOP=[], NPR=NPR, RL1=200.0, RL0=-200.0)

# 2D continuations
##################

NPR = 20

# 2D continuation follow-up in Delta and eta for d = 10
a.run(starting_point='LP1', c='qif2', ICP=[5, 8], name='Delta/eta:lp1', origin=f'eta:1', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[5, 8], name='Delta/eta:lp2', origin=f'eta:1', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)

# 2D continuation follow-up in Delta and eta for d = 100
a.run(starting_point='LP1', c='qif2', ICP=[5, 8], name='Delta/eta:lp3', origin=f'eta:2', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)
a.run(starting_point='LP2', c='qif2', ICP=[5, 8], name='Delta/eta:lp4', origin=f'eta:2', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)
a.run(starting_point='HB1', c='qif2', ICP=[5, 8], name='Delta/eta:hb1', origin=f'eta:2', NMX=8000, DSMAX=0.05,
      NPR=NPR, RL1=100.0, RL0=0.0, bidirectional=True)

# plotting
##########

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (4, 2)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6


# plot 2D continuations
fig, axes = plt.subplots(ncols=2)

# d = 10
ax = axes[0]
a.plot_continuation(f'PAR(8)', f'PAR(5)', cont=f'Delta/eta:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E')
a.plot_continuation(f'PAR(8)', f'PAR(5)', cont=f'Delta/eta:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E')
ax.set_xlabel(r"$\eta$ (pA)")
ax.set_ylabel(r"$\Delta_v$ (mV)")
ax.set_title(r"$\kappa = 10$")
ax.set_xlim([20.0, 70.0])
ax.set_ylim([0.0, 4.0])
ax.axhline(y=0.2, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axhline(y=1.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axhline(y=2.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=55.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)

# d = 100
ax = axes[1]
a.plot_continuation(f'PAR(8)', f'PAR(5)', cont=f'Delta/eta:lp3', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable="solid")
a.plot_continuation(f'PAR(8)', f'PAR(5)', cont=f'Delta/eta:lp4', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E')
a.plot_continuation(f'PAR(8)', f'PAR(5)', cont=f'Delta/eta:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77')
ax.set_xlabel(r"$\kappa$ (pA)")
ax.set_ylabel(r"$\Delta_v$ (mV)")
ax.set_title(r"$\kappa = 100$")
ax.set_xlim([20.0, 70.0])
ax.set_ylim([0.0, 1.6])
ax.axhline(y=0.1, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axhline(y=1.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)
ax.axvline(x=55.0, color='black', alpha=0.5, linestyle='--', linewidth=0.5)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'../results/rs_bifs.svg')
plt.show()
