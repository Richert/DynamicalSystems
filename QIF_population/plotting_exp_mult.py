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
# data loading #
################

# load pyauto instance from file
fname = 'results/exp_mult.pkl'
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
for pd in a.additional_attributes['pd_solutions']:
    ax = a.plot_continuation('PAR(1)', 'U(1)', cont=pd, ax=ax, ignore=['LP', 'BP'], default_size=markersize1)

# visualization of phase portrait
#################################

# # 3d plot of phase portrait for eta, alpha and e
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
for cm, pd in zip(cmaps, a.additional_attributes['pd_solutions'][:-2]):
    #ax2 = a.plot_trajectory(vars=['U(1)', 'U(2)', 'U(3)'], cont=pd, ax=ax2, linewidths=2.0, point='PD1', # for some reason they are numbered pd_1 for me -- 
    #                     cmap=plt.get_cmap(cm), force_axis_lim_update=True)
    ax2 = a.plot_trajectory(vars=['U(1)', 'U(2)', 'U(3)'], cont=a.additional_attributes['pd_solutions'][-1], ax=ax2,
                         linewidths=2.0, point='PD1', cmap=plt.get_cmap('magma'))

# visualization of codim 2 bifurcations
#######################################

fig, ax = plt.subplots(figsize=(2, 2), dpi=dpi)

# plot eta-alpha continuation of the limit cycle
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont=f'eta_alpha_hb2', ax=ax, ignore=['LP', 'BP'])

# plot eta-alpha continuation of the period doubling bifurcation
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='eta_alpha_pd', ax=ax, ignore=['LP', 'BP'])



# visualization of chaos measures (extracted during period doubling bifurcations)
#################################

chaos_data = a.additional_attributes['chaos_analysis_pd']
fig, axes = plt.subplots(ncols=2, figsize=(7, 1.8), dpi=dpi)
for i, key in enumerate(chaos_data):

    x = np.asarray(chaos_data[key]['eta'])
    fd = np.asarray(chaos_data[key]['fractal_dim'])
    lps = chaos_data[key]['lyapunov_exponents']
    idx = [i for i in range(len(lps)) if lps[i]]
    lp_max = np.max(np.asarray([lps[i] for i in idx]), axis=1)

    axes[0].plot(x[idx], lp_max)
    axes[1].plot(x[idx], fd[idx])

axes[0].set_title('Max LP')
axes[1].set_title('FD')

plt.tight_layout()
plt.show()



# load pyauto instance from file
fname = 'results/exp_mult_strange_attractor.pkl'
a = PyAuto.from_file(fname)

chaos_data = a.additional_attributes['chaos_analysis']

fig = plt.figure()


etas = chaos_data['eta']
eta_rounded = list(map(lambda x : round(x,2), chaos_data['eta'])) 
lps = chaos_data['lyapunov_exponents']
fract_dim = chaos_data['fractal_dim']

n_per_row = 4
n_rows = int(np.ceil(len(etas)/n_per_row))
gs = fig.add_gridspec(n_rows, n_per_row)

row = 0
col = 0
for i, elem in enumerate(eta_rounded):
    f_ax1 = fig.add_subplot(gs[row, col], projection='3d')
    ax = a.plot_trajectory(['U(1)', 'U(2)', 'U(3)'], ax=f_ax1, cont=f'eta_{i+1}_t', force_axis_lim_update=True, cmap=plt.get_cmap('magma'),cutoff=200)
    max_LP = round(max(chaos_data['lyapunov_exponents'][i]), 3)
    fract_dim = round(chaos_data['fractal_dim'][i], 3)
    ax.set_title(fr'$\bar \eta$ ={elem}'+f'\nMax LP: {max_LP}'+f'\nDim: {fract_dim}')
    col+=1
    if not (i+1)%n_per_row:
        row+=1
        col=0

imagepath='../../plots/'+f'exp_mult_{eta_rounded}'+'.png'
plt.savefig(imagepath)
plt.show()
