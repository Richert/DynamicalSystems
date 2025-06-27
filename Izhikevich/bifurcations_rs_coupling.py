from pycobi import ODESystem
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

# config
n_dim = 4
n_params = 20
a = ODESystem("rs2", working_dir="config", auto_dir=auto_dir, init_cont=False)

# initial continuation in time to converge to fixed point
t_sols, t_cont = a.run(c='ivp', name='t', DS=1e-4, DSMIN=1e-10, EPSL=1e-06, NPR=1000, NPAR=n_params, NDIM=n_dim,
                       EPSU=1e-06, EPSS=1e-05, DSMAX=0.1, NMX=50000, UZR={14: 500.0}, STOP={'UZ1'})

########################
# bifurcation analysis #
########################

# prepare state
###############

# continuation in d
c0_sols, c0_cont = a.run(starting_point='UZ1', c='qif', ICP=17, NPAR=n_params, NDIM=n_dim, name='d:1',
                         origin=t_cont, NMX=8000, DSMAX=0.1, UZR={17: [80.0]}, STOP=[f'UZ1'], NPR=100,
                         RL1=350.0, RL0=0.0)

# continuation in Delta
vals = [0.1, 0.8, 1.6]
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=5, NPAR=n_params, NDIM=n_dim, name='D:1',
                         origin=c0_cont, NMX=8000, DSMAX=0.05, UZR={5: vals}, STOP=[f'UZ{len(vals)}'], NPR=100,
                         RL1=3.0, RL0=0.0, bidirectional=True)

# main continuations
####################

# continuations in I for Delta = 0.1
r1_sols, r1_cont = a.run(starting_point='UZ1', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='I_ext:1',
                         origin=c1_cont, NMX=8000, DSMAX=0.05, UZR={}, STOP=[], NPR=100,
                         RL1=150.0, RL0=0.0, bidirectional=True)

# continuations in I for Delta = 0.5
r2_sols, r2_cont = a.run(starting_point='UZ2', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='I_ext:2',
                         origin=c1_cont, NMX=8000, DSMAX=0.05, UZR={}, STOP=[], NPR=100,
                         RL1=150.0, RL0=0.0, bidirectional=True)
# a.run(starting_point='LP1', c='qif2', ICP=[5, 8], name='D/I:lp1', origin=r1_cont, NMX=8000, DSMAX=0.05,
#       NPR=20, RL1=5.0, RL0=0.0, bidirectional=True)
# a.run(starting_point='LP2', c='qif2', ICP=[5, 8], name='D/I:lp2', origin=r1_cont, NMX=8000, DSMAX=0.05,
#       NPR=20, RL1=5.0, RL0=0.0, bidirectional=True)

# continuations in I for Delta = 1.0
r3_sols, r3_cont = a.run(starting_point='UZ3', c='qif', ICP=8, NPAR=n_params, NDIM=n_dim, name='I_ext:3',
                         origin=c1_cont, NMX=8000, DSMAX=0.05, UZR={}, STOP=[], NPR=100,
                         RL1=150.0, RL0=0.0, bidirectional=True)

# plotting
##########

# create figure layout
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(nrows=3, ncols=1)

# plot the 2D bifurcation diagram
# ax = fig.add_subplot(grid[:, 0])
# a.plot_continuation('PAR(16)', 'PAR(8)', cont=f'v_0/I:lp1', ax=ax, line_color_stable='#5D6D7E',
#                     line_color_unstable='#5D6D7E', line_style_unstable='solid')
# a.plot_continuation('PAR(16)', 'PAR(8)', cont=f'v_0/I:lp2', ax=ax, line_color_stable='#5D6D7E',
#                     line_color_unstable='#5D6D7E', line_style_unstable='solid')
# ax.set_xlabel(r'$I$')
# ax.set_ylabel(r'$v_0$')
# ax.set_title('(A) 2D bifurcation diagram')
# ax.set_xlim([-20.0, 60.0])
# ax.set_ylim([-160.0, -60.0])

# plot the 1D bifurcation diagrams
for i, D in enumerate(vals):

    ax = fig.add_subplot(grid[i, 0])
    a.plot_continuation('PAR(8)', 'U(4)', cont=f'I_ext:{i+1}', ax=ax)
    ax.set_xlabel(r'$I$')
    ax.set_ylabel(r'$s$')
    ax.set_title(rf'$\Delta = {D}$')

# padding
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.01, hspace=0.05, wspace=0.)

# saving/plotting
fig.canvas.draw()
# plt.savefig(f'results/rs_corrected.pdf')
plt.show()

