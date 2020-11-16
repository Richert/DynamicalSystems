import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pyrates.utility.pyauto import PyAuto
import sys
sys.path.append('../')


# plotting parameters
linewidth = 1.2
fontsize1 = 12
fontsize2 = 12
markersize1 = 120
markersize2 = 100
dpi = 200
plt.style.reload_library()
plt.style.use('seaborn-whitegrid')
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
#mpl.rc('text', usetex=True)
mpl.rcParams["font.sans-serif"] = ["Roboto"]
mpl.rcParams["font.size"] = fontsize1
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['axes.titlesize'] = fontsize2
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['ytick.alignment'] = 'center'
mpl.rcParams['legend.fontsize'] = fontsize1


################
# file loading #
################

fname = 'results/tsodyks_poisson.pkl'
a = PyAuto.from_file(fname, auto_dir='~/PycharmProjects/auto-07p')

############
# plotting #
############

# principle continuation in eta
###############################

fig, axes = plt.subplots(ncols=2, figsize=(8.6, 3.1), dpi=dpi)

# plot principle eta continuation for different alphas
ax = axes[0]
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta', ax=ax, default_size=markersize1)
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_hb1', ax=ax, ignore=['UZ', 'BP'], default_size=markersize2)
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_hb2', ax=ax, ignore=['UZ', 'BP', 'LP'], default_size=markersize2)
ax.set_xlabel(r'$\bar\eta$')
ax.set_ylabel('r')
ax.set_xlim([-1.0, -0.7])
ax.set_ylim([0.0, 0.5])

# plot eta continuation for single alpha with limit cycle continuation
ax = axes[1]
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_2', ax=ax, default_size=markersize1)
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_2_hb1', ax=ax, ignore=['UZ', 'BP', 'LP'], default_size=markersize2)
ax.set_xlabel(r'$\bar\eta$')
ax.set_ylabel('r')
ax.set_xlim([-1.0, 2.0])
ax.set_ylim([-0.2, 2.0])

plt.tight_layout()
plt.savefig('tsodyks_1d.svg')

# 2D continuations in eta and (alpha, U0)
#########################################

fig2, axes2 = plt.subplots(ncols=2, figsize=(8.6, 3.1), dpi=dpi)
ax = axes2[0]
# a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_3', ax=ax, default_size=markersize1)
# ax.set_xlabel(r'$\bar\eta$')
# ax.set_ylabel('r')
# ax.set_xlim([-1.0,-0.2])
# ax.set_ylim([0.0, 0.6])
a.plot_continuation('PAR(1)', 'PAR(7)', cont='eta_Delta_lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_style_stable='dashed', line_style_unstable='dashed', default_size=markersize2)
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\Delta$')
ax = axes2[1]
a.plot_continuation('PAR(1)', 'PAR(7)', cont='eta_Delta_hb1', ax=ax, line_style_unstable='solid',
                    default_size=markersize2)
a.plot_continuation('PAR(1)', 'PAR(7)', cont='eta_Delta_lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_style_stable='dashed', line_style_unstable='dashed', default_size=markersize2)
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\Delta$')
ax.set_xlim([-1.0, 1.0])
ax.set_ylim([0.0, 0.55])

plt.tight_layout()
plt.savefig('tsodyks_2d.svg')

plt.show()
