import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from matplotlib.colors import to_hex
import pandas as pd
from pyrates.utility.pyauto import PyAuto
import sys
sys.path.append('../')


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


# preparations
##############

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (3.5, 3.5)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.6
plt.rcParams['axes.titlepad'] = 1.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 15
cmap = sns.color_palette("plasma", as_cmap=False, n_colors=100)

# load auto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
fname = '../QIF_population/results/alpha_mult.pkl'
a = PyAuto.from_file(fname, auto_dir=auto_dir)
a.update_bifurcation_style(bf_type='HB', color='k')

# plotting
##########

# codim 2 bifurcations
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=1, ncols=10, figure=fig)

# create data frame with codim 2 periods
alphas = np.round(np.linspace(0., 0.2, 100)[::-1], decimals=3)
etas = np.round(np.linspace(-6.5, -2.5, 100), decimals=2)
vmax = 100.0
data = a.additional_attributes['tau_e_p_periods']
data[data > vmax] = vmax
df = pd.DataFrame(interpolate2d(data), index=alphas, columns=etas)

# plot codim 2 periods
# ax = a.plot_heatmap(df, ax=ax, cmap='magma', cbar_ax=axes2[2], mask=df.values == 0)
#
# cosmetics
# yticks = [label._y for label in ax.get_yticklabels()]
# yticklabels = [float(label._text) for label in ax.get_yticklabels()]
# xticks = [label._x for label in ax.get_xticklabels()]
# xticklabels = [float(label._text) for label in ax.get_xticklabels()]
# ax.set_xticks(xticks[::3])
# ax.set_xticklabels(xticklabels[::3])
# ax.set_yticks(yticks[::3])
# ax.set_yticklabels(yticklabels[::3])
# ax.set_xlabel(r'$\bar\eta$')
# ax.set_ylabel(r'$\alpha$')
# ax.set_title('Limit Cycle Period')

ax = fig.add_subplot(grid[0, :-1])

# plot period image
cbar_ax = fig.add_subplot(grid[0, -1])
ax = a.plot_heatmap(df, ax=ax, cmap=cmap, cbar_ax=cbar_ax, mask=df.values == 0)

# plot eta-alpha continuation of the limit cycle
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont=f'eta_alpha_hb2', ax=ax, ignore=['LP', 'BP'],
                         line_style_unstable='dashed', default_size=markersize)

# plot eta-alpha continuation of the limit cycle fold bifurcations
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='eta_alpha_lp2', ax=ax, ignore=['LP', 'BP', 'UZ'],
                         line_color_stable='#148F77', line_color_unstable='#148F77', line_style_unstable='dotted',
                         default_size=markersize)
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='eta_alpha_lp3', ax=ax, ignore=['LP', 'BP', 'UZ'],
                         line_color_stable='#148F77', line_color_unstable='#148F77', line_style_unstable='dotted',
                         default_size=markersize)

# plot eta-alpha continuation of the fold bifurcation
ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='eta_alpha_lp1', ax=ax, ignore=['LP', 'BP', 'UZ'],
                         line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E', line_style_unstable='solid',
                         default_size=markersize)

# cosmetics
ax.set_xlabel(r'$\bar\eta$')
ax.set_ylabel(r'$\alpha$')
# ax.set_xticks(xticklabels[::3])
# ax.set_yticks(yticklabels[::3])
# ax.set_xlim([-6.5, -2.5])
# ax.set_ylim([0., 0.2])
ax.set_title('2D Limit Cycle Continuation')

# final touches
###############

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving
fig.canvas.draw()
plt.savefig(f'neco_sd_2d_bf.svg')
plt.show()
