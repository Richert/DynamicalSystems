import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pyrates.utility.pyauto import PyAuto


# plotting parameters
linewidth = 0.5
fontsize1 = 6
fontsize2 = 8
markersize1 = 25
markersize2 = 25
dpi = 400
plt.style.reload_library()
plt.style.use('seaborn-whitegrid')
mpl.rcParams['font.family'] = 'Roboto'
mpl.rcParams['font.size'] = fontsize1
mpl.rcParams['lines.linewidth'] = linewidth
#mpl.rcParams['ax_data.titlesize'] = fontsize2
#mpl.rcParams['ax_data.titleweight'] = 'bold'
#mpl.rcParams['ax_data.labelsize'] = fontsize2
#mpl.rcParams['ax_data.labelcolor'] = 'black'
#mpl.rcParams['ax_data.labelweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = fontsize1
mpl.rcParams['ytick.labelsize'] = fontsize1
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['legend.fontsize'] = fontsize1
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
mpl.rc('text', usetex=True)


################
# file loading #
################

fname = 'results/tsodyks.hdf5'
a = PyAuto.from_file(fname)

############
# plotting #
############

# principle continuation in eta
###############################

fig, axes = plt.subplots(ncols=2, figsize=(7, 1.8), dpi=dpi)

# plot principle eta continuation for different alphas
ax = axes[0]
n_alphas = 6
for i in range(n_alphas):
    try:
        ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_{i}', ax=ax)
    except KeyError:
        pass
ax.set_xlabel(r'$\eta$')

# plot eta continuation for single alpha with limit cycle continuation
ax = axes[1]
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_4', ax=ax)
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta/hb1', ax=ax, ignore=['UZ', 'BP'])
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta/hb2', ax=ax, ignore=['UZ', 'BP'])
ax.set_xlabel(r'$\eta$')

plt.tight_layout()
plt.show()