import matplotlib.pyplot as plt
import matplotlib as mpl
from pyauto import PyAuto
import numpy as np
import pandas as pd

# plotting parameters
linewidth = 0.5
fontsize1 = 6
fontsize2 = 8
markersize1 = 20
markersize2 = 20
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
#mpl.rcParams['xtick.labelsize'] = fontsize1
#mpl.rcParams['ytick.labelsize'] = fontsize1
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['legend.fontsize'] = fontsize1
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
mpl.rc('text', usetex=True)


################
# file loading #
################

fname = 'results/alpha_mult.pkl'
a = PyAuto.from_file(fname)

############
# plotting #
############

# principle continuation in eta
###############################

fig, axes = plt.subplots(ncols=2, figsize=(7, 1.8), dpi=dpi)

# plot principle eta continuation for different alphas
ax = axes[0]
n_alphas = 5
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_0', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
for i in range(1, n_alphas+1):
    ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_{i}', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
ax.set_xlim([-12.0, -2.0])
ax.set_ylim([0., 1.2])
ax.set_xlabel(r'$\eta$')
ax.set_ylabel('firing rate (r)')
ax.set_title('Effects of Adaptation Rate')

# plot eta continuation for single alpha with limit cycle continuation
ax = axes[1]
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_3', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_hb2', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1)
ax.set_xlim([-6.5, -4.0])
ax.set_ylim([0., 2.5])
ax.set_xlabel(r'$\eta$')
ax.set_ylabel('firing rate (r)')
ax.set_title(r'Principal Continuation in $\eta$')
plt.tight_layout()

# 2D continuation in eta and alpha
##################################

# codim 2 bifurcations
fig2, axes2 = plt.subplots(figsize=(1.8, 4.5), dpi=dpi, nrows=2)

# create data frame with codim 2 periods
alphas = np.round(np.linspace(0., 0.2, 100)[::-1], decimals=2)
etas = np.round(np.linspace(-6.5, -2.5, 100), decimals=2)
df = pd.DataFrame(a.period_solutions, index=alphas, columns=etas)

# plot codim 2 periods
ax = axes2[1]
ax = a.plot_heatmap(df, ax=ax)

# cosmetics
yticks = [label._y for label in ax.get_yticklabels()]
yticklabels = [float(label._text) for label in ax.get_yticklabels()]
xticks = [label._x for label in ax.get_xticklabels()]
xticklabels = [float(label._text) for label in ax.get_xticklabels()]
ax.set_xticks(xticks[::3])
ax.set_xticklabels(xticklabels[::3])
ax.set_yticks(yticks[::3])
ax.set_yticklabels(yticklabels[::3])
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\alpha$')
ax.set_title('2D Limit Cycle Continuation')

# plot eta-alpha continuation of the limit cycle
ax = axes2[0]
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont=f'eta_alpha_hb2', ax=ax, ignore=['LP', 'BP'],
                         line_style_unstable='solid', default_size=markersize1)

# plot eta-alpha continuation of the limit cycle fold bifurcations
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='eta_alpha_lp2', ax=ax, ignore=['LP', 'BP', 'UZ', 'R1'],
                         line_color_stable='#148F77', line_color_unstable='#148F77', line_style_unstable='solid',
                         default_size=markersize1)
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='eta_alpha_lp3', ax=ax, ignore=['LP', 'BP', 'UZ', 'R1'],
                         line_color_stable='#148F77', line_color_unstable='#148F77', line_style_unstable='solid',
                         default_size=markersize1)

# plot eta-alpha continuation of the fold bifurcation
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='eta_alpha_lp1', ax=ax, ignore=['LP', 'BP', 'UZ'],
                         line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E', line_style_unstable='solid',
                         default_size=markersize1)

# cosmetics
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\alpha$')
ax.set_xticks(xticklabels[::3])
ax.set_yticks(yticklabels[::3])
ax.set_xlim([-6.5, -2.5])
ax.set_ylim([0., 0.2])
ax.set_title('Limit Cycle Period')

plt.tight_layout()
plt.show()
