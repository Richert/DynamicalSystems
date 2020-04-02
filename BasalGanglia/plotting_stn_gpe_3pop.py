import matplotlib.pyplot as plt
import matplotlib as mpl
from pyauto import PyAuto
import numpy as np
import pandas as pd

# plotting parameters
# linewidth = 2.0
# fontsize1 = 12
# fontsize2 = 14
# markersize1 = 80
# markersize2 = 80
linewidth = 2.0
fontsize1 = 14
fontsize2 = 16
markersize1 = 100
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

fname = 'results/stn_gpe_3pop.pkl'
a = PyAuto.from_file(fname)

############
# plotting #
############

# principle continuation in dopa
################################

fig, axes = plt.subplots(figsize=(4.0, 6.0), dpi=dpi, nrows=2)

# plot eta continuation for single alpha with limit cycle continuation
ax = axes[0]
ax = a.plot_continuation('PAR(1)', 'U(2)', cont='fp', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(1)', 'U(2)', cont='lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlabel(r'dopamine depletion')
ax.set_ylabel('Firing rate (STN)')
ax.set_title(r'Bursting Limit Cycle')
plt.tight_layout()

# plot eta continuation for single alpha with limit cycle continuation
ax = axes[1]
ax = a.plot_continuation('PAR(1)', 'PAR(16)', cont='2d1', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlabel(r'dopamine depletion')
ax.set_ylabel(r'$\Delta_i$')
ax.set_title(r'Codim 2 Bifurcations of HB1')
plt.tight_layout()
plt.show()
