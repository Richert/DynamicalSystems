import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pyrates.utility.pyauto import PyAuto
import random

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

# load pyauto instance from file (from script bifurcation_exp_mult.py)
fname = 'results/exp_mult.pkl'
a = PyAuto.from_file(fname)

# load pyauto instance from file (time continuation from script bifurcation_exp_mult_time.py)
fname = 'results/exp_mult_time.pkl'
b = PyAuto.from_file(fname)
chaos_data = b.additional_attributes['chaos_analysis']
etas = chaos_data['eta']

eta_rounded = list(map(lambda x : round(x,3), chaos_data['eta'])) 
lps = chaos_data['lyapunov_exponents']
fract_dim = chaos_data['fractal_dim']

if len(eta_rounded)>3:
    indices_to_select=[]
    three_etas = np.squeeze(np.unique(eta_rounded))[::-1]
    for i in range(3):
        indices_same = np.squeeze(np.where(eta_rounded==three_etas[i]))
        try: # if there are more continuations of the same eta
            indices_to_select.append((random.choice(indices_same)))
        except TypeError: # otherwise we just select the one 
            indices_to_select.append(indices_same)

    eta_rounded = [eta_rounded[i] for i in indices_to_select]
    lps = [lps[i] for i in indices_to_select]
    fract_dim = [fract_dim[i] for i in indices_to_select]
else:
    indices_to_select=range(3)

#################
# visualization #
#################

#fig, axes = plt.subplots(ncols=2, figsize=(7, 1.8), dpi=dpi)

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(3, 3)

# plot eta continuation
#######################
f_ax1 = fig.add_subplot(gs[:, 0])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# plot eta continuation for single alpha with limit cycle continuation
a.update_bifurcation_style('PD', color='k')
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_3', ax=f_ax1, default_size=markersize1)
for pd in a.additional_attributes['pd_solutions']:
    ax = a.plot_continuation('PAR(1)', 'U(1)', cont=pd, ax=ax, ignore=['BP'], default_size=markersize1)
    ax.set_xlim((-6, -5))
for i, eta in enumerate(eta_rounded):
    ax.axvline(eta,0,1, color=colors[i], linewidth=1)
ax.set_xlabel(r'$\bar\eta$')
ax.set_ylabel('firing rate (r)')


cutoff=300
for i, eta in enumerate(eta_rounded):
    f_ax2 = fig.add_subplot(gs[i, 1])
    ax = b.plot_continuation('PAR(14)', 'U(1)', cont=f'eta_{indices_to_select[i]+1}_t', ax=f_ax2)
    bottom, top = ax.get_xlim()
    ax.set_xlim((cutoff, top))
    ax.set_xlabel('time')
    ax.set_ylabel('firing rate (r)')
    ax.set_title(fr'$\bar\eta$ ={eta}', color=colors[i])


for i, eta in enumerate(eta_rounded):
    f_ax3 = fig.add_subplot(gs[i, 2], projection='3d')
    max_LP = round(max(lps[i]), 3)
    fract_dimen = round(fract_dim[i], 3)
    ax=b.plot_trajectory(['U(1)', 'U(2)', 'U(3)'], ax=f_ax3, cont=f'eta_{indices_to_select[i]+1}_t', force_axis_lim_update=True, cutoff=cutoff, cmap=plt.get_cmap('magma'))
    ax.set_title(fr'$\bar\eta$ ={eta}'+f'\nMax LP: {max_LP}'+f'\nDim: {fract_dimen}', color=colors[i])


imagepath='../../plots/'+f'bif_exp_mult_time_{eta_rounded}'+'.pdf'
plt.savefig(imagepath)
plt.show()
