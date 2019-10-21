import matplotlib.pyplot as plt
import matplotlib as mpl
from pyauto import PyAuto
import numpy as np
import pandas as pd

# plotting parameters
linewidth = 2.0
linewidth2 = 2.0
fontsize1 = 12
fontsize2 = 14
markersize1 = 100
markersize2 = 100
dpi = 500

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


def interpolate2d(x, val=0.0):
    m, n = x.shape
    for i in range(m):
        for j in range(n):
            if i != 0 and j != 0 and i != m-1 and j != n-1:
                vals = np.asarray([x[i-1, j-1], x[i, j-1], x[i+1, j-1], x[i-1, j], x[i+1, j], x[i-1, j+1], x[i, j+1],
                                   x[i+1, j+1]])
                if np.sum(vals == val) < 3 and x[i, j] < np.mean(vals):
                    x[i, j] = np.mean(vals[vals != val])
    return x

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

fig, axes = plt.subplots(ncols=2, figsize=(13.4, 3.0), dpi=dpi)

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
ax.set_ylabel('Firing rate (r)')
ax.set_title('Fixed Points')

# plot eta continuation for single alpha with limit cycle continuation
ax = axes[1]
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_3', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_hb2', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         default_size=markersize1, custom_bf_styles={'LP': {'marker': 'p'}})
ax.set_xlim([-6.5, -3.8])
ax.set_ylim([0., 2.5])
ax.set_xlabel(r'$\eta$')
ax.set_ylabel('Firing rate (r)')
ax.set_title(r'Bursting Limit Cycle')
plt.tight_layout()
plt.savefig('fig1.svg')

# 2D continuation in eta and alpha
##################################

# codim 2 bifurcations
fig2, axes2 = plt.subplots(figsize=(14.5, 7), dpi=dpi, ncols=3, gridspec_kw={'width_ratios': [7/15, 7/15, 1/30]})

# create data frame with codim 2 periods
alphas = np.round(np.linspace(0., 0.2, 100)[::-1], decimals=3)
etas = np.round(np.linspace(-6.5, -2.5, 100), decimals=2)
vmax = 100.0
data = a.period_solutions
data[data > vmax] = vmax
df = pd.DataFrame(interpolate2d(data), index=alphas, columns=etas)

# plot codim 2 periods
ax = axes2[1]
ax = a.plot_heatmap(df, ax=ax, cmap='magma', cbar_ax=axes2[2], mask=df.values == 0)

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
ax.set_title('Limit Cycle Period')


# plot eta-alpha continuation of the limit cycle

ax = axes2[0]
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont=f'eta_alpha_hb2', ax=ax, ignore=['LP', 'BP'],
                         line_style_unstable='dashed', default_size=markersize1)

# plot eta-alpha continuation of the limit cycle fold bifurcations
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='eta_alpha_lp2', ax=ax, ignore=['LP', 'BP', 'UZ', 'R1'],
                         line_color_stable='#148F77', line_color_unstable='#148F77', line_style_unstable='dotted',
                         default_size=markersize1, linewidth=linewidth2)
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='eta_alpha_lp3', ax=ax, ignore=['LP', 'BP', 'UZ', 'R1'],
                         line_color_stable='#148F77', line_color_unstable='#148F77', line_style_unstable='dotted',
                         default_size=markersize1, linewidth=linewidth2)

# plot eta-alpha continuation of the fold bifurcation
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='eta_alpha_lp1', ax=ax, ignore=['LP', 'BP', 'UZ'],
                         line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E', line_style_unstable='solid',
                         default_size=markersize1)

# transfer line plots to period image

# cosmetics
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\alpha$')
ax.set_xticks(xticklabels[::3])
ax.set_yticks(yticklabels[::3])
ax.set_xlim([-6.5, -2.5])
ax.set_ylim([0., 0.2])
ax.set_title('2D Limit Cycle Continuation')
plt.tight_layout()
plt.savefig('fig2.svg')

plt.show()
