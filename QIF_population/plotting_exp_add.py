import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pyauto import PyAuto

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
# data loading #
################

# load pyauto instance from file
fname = 'results/exp_add.hdf5'
a = PyAuto.from_file(fname)

#################
# visualization #
#################

fig, axes = plt.subplots(ncols=2, figsize=(7, 1.8), dpi=dpi)

# plot eta continuation
#######################

# plot principle eta continuation for different alphas
ax = axes[0]
n_alphas = 5
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_0', ax=ax, default_size=markersize1)
for i in range(1, n_alphas+1):
    ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_{i}', ax=ax, default_size=markersize1)

# plot eta continuation for single alpha with limit cycle continuation
ax = axes[1]
a.update_bifurcation_style('PD', color='k')
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_3', ax=ax, default_size=markersize1)
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_hb2', ax=ax, ignore=['LP', 'BP'], default_size=markersize1)
for i, pd in enumerate(a.pd_solutions[:-2]):
    ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'pd_{i}', ax=ax, ignore=['LP', 'BP'], default_size=markersize1)

# visualization of phase portrait
#################################

# 3d plot of phase portrait for eta, alpha and e
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
for cm, pd in zip(cmaps, a.pd_solutions[:-2]):
    ax2 = a.plot_trajectory(vars=['U(1)', 'U(2)', 'U(3)'], cont=pd, ax=ax2, linewidths=2.0, point='PD1',
                            cmap=plt.get_cmap(cm), force_axis_lim_update=True)
ax2 = a.plot_trajectory(vars=['U(1)', 'U(2)', 'U(3)'], cont=a.pd_solutions[-1], ax=ax2, linewidths=2.0, point='PD1',
                        cmap=plt.get_cmap('magma'))

# visualization of codim 2 bifurcations
#######################################

fig, ax = plt.subplots(figsize=(2, 2), dpi=dpi)

# plot eta-alpha continuation of the limit cycle
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont=f'eta_alpha_hb2', ax=ax, ignore=['LP', 'BP'])

# plot eta-alpha continuation of the period doubling bifurcation
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='eta_alpha_pd', ax=ax, ignore=['LP', 'BP'])

plt.show()
