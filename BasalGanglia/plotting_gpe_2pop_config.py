from pyrates.utility.pyauto import PyAuto
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.colors import to_hex
import seaborn as sns
import sys
import numpy as np
import pandas as pd
import pickle

"""
Basic input-output curve analysis of two GPe populations (arkypallidal and prototypical) without synapses.
Creates Fig. 1 of Gast et al. (2021) JNS.

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) and 
auto-07p (https://github.com/auto-07p/auto-07p) installed.
You can pass your auto-07p installation path to the python command ('python gpe_bifurcation_analysis.py custom_path') 
or change the default value of `auto_dir` below.
"""

# calculate single-cell firing rate distributions
#################################################

# general QIF population parameters
N = 1000

# gpe-p parameters
tau_p = 18.0
Delta_p = 9.0
eta_p = 11.0
eta_p_distr = eta_p+Delta_p * np.tan((np.pi/2.0)*(2.0*np.arange(0, N)-N)/(N+1))

# gpe-a parameters
tau_a = 32.0
Delta_a = 3.0
eta_a = 0.5
eta_a_distr = eta_a+Delta_a * np.tan((np.pi/2.0)*(2.0*np.arange(0, N)-N)/(N+1))

# calculate firing rate distribution of gpe-p neurons
rates_p = np.zeros_like(eta_p_distr)
rates_p[eta_p_distr >= 0] = 1e3*np.sqrt(eta_p_distr[eta_p_distr >= 0])/(tau_p*np.pi)

# calculate firing rate distribution of gpe-a neurons
rates_a = np.zeros_like(eta_a_distr)
rates_a[eta_a_distr >= 0] = 1e3*np.sqrt(eta_a_distr[eta_a_distr >= 0])/(tau_a*np.pi)

# add firing rates to a dataframe
df = pd.DataFrame(data=[[0.0, 0.0, '']], columns=['r', 'eta', 'population'])
df = df.append(pd.DataFrame(data=[[rates_p[i], eta_p_distr[i], 'GPe-p'] for i in range(N)],
                            columns=['r', 'eta', 'population']))
df = df.append(pd.DataFrame(data=[[rates_a[i], eta_a_distr[i], 'GPe-a'] for i in range(N)],
                            columns=['r', 'eta', 'population']))
df = df.iloc[1:, :]
df.index = np.arange(0, 2*N)

# calculate input-output curves of gpe-p and gpe-a populations
##############################################################

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 20
n_params = 21
a = PyAuto("auto_files", auto_dir=auto_dir)

# initial continuation in time
t_sols, t_cont = a.run(e='gpe_2pop', c='ivp', ICP=14, DS=5e-3, DSMIN=1e-4, DSMAX=1.0, NMX=1000000, name='t',
                       UZR={14: 10000.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

# continuation of eta_p for k_gp = 0.0
c1_sols, c1_cont = a.run(starting_point='UZ1', c='qif', ICP=2, NPAR=n_params, name='eta_p:1', NDIM=n_dim, RL0=-100,
                         RL1=100.0, origin=t_cont, NMX=8000, DSMAX=0.05, STOP={}, bidirectional=True)

# continuation of eta_a for k_gp = 0.0
c2_sols, c2_cont = a.run(starting_point='UZ1', c='qif', ICP=3, NPAR=n_params, name='eta_a:1', NDIM=n_dim, RL0=-100,
                         RL1=100.0, origin=t_cont, NMX=8000, DSMAX=0.05, STOP={}, bidirectional=True)

# load average firing rate data of gpe-p and gpe-a
##################################################

rates = pickle.load(open(f"../results/gpe_2pop_config.p", "rb"))['rates']

# plotting
##########

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (4.5, 4.0)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['axes.titlepad'] = 0.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 25
cmap = sns.color_palette("plasma", as_cmap=False, n_colors=4)

# create figure layout
fig = plt.figure(1)
grid = gs.GridSpec(nrows=3, ncols=2, figure=fig)

# firing rate distributions
ax1 = fig.add_subplot(grid[0, :])
sns.histplot(data=df, x='r', hue='population', ax=ax1, log_scale=False, kde=False, bins=int(N/10),
             palette=[cmap[0], cmap[-1]], stat='probability')
ax1.set_xlim([0, 150])
ax1.set_title('A')

# QIF activation functions
ax2 = fig.add_subplot(grid[1, 0])
ax2.plot(eta_p_distr-eta_p, rates_p, c=cmap[0])
ax2.plot(eta_a_distr-eta_a, rates_a, c=cmap[-1])
ax2.set_xlabel(r'$\eta_i$')
ax2.set_ylabel(r'$r_i$')
ax2.set_title('B: QIF neuron I-O curves')
ax2.set_xlim([-50.0, 100.0])
ax2.set_ylim([0.0, 200.0])
ax2.set_yticks([0, 60, 120, 180])
plt.legend(['GPe-p', 'GPe-a'])

# population activation functions
ax3 = fig.add_subplot(grid[1, 1])
ax3 = a.plot_continuation('PAR(2)', 'U(2)', cont='eta_p:1', ax=ax3, line_color_stable=to_hex(cmap[0]))
ax3 = a.plot_continuation('PAR(3)', 'U(4)', cont='eta_a:1', ax=ax3, line_color_stable=to_hex(cmap[-1]))
ax3.set_xlabel(r'$\bar \eta$')
ax3.set_ylabel(r'$r$')
ax3.set_title('Population I-O curves')
ax3.set_xlim([-50.0, 100.0])
ax3.set_ylim([0.0, 0.2])
ax3.set_yticks([0, 0.06, 0.12, 0.18])
ax3.set_yticklabels(['0', '60', '120', '180'])
plt.legend(['GPe-p', 'GPe-a'])

# average firing rates across conditions
ax4 = fig.add_subplot(grid[2, :])
sns.barplot(data=rates, x='condition', y='r', hue='population', ax=ax4, palette=[cmap[0], cmap[-1]], ci=None)
ax4.set_title('C')
ax4.set_yticks([0, 30, 60, 90])

# changes of figure layout
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)

# save figure
fig.canvas.draw()
plt.savefig(f'../results/gpe_2pop_config.svg')

plt.show()
