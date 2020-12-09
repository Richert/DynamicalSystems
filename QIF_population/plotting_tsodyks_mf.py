import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
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

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

fname = 'results/tsodyks_mf.pkl'
fname2 = 'results/tsodyks_poisson.pkl'
a = PyAuto.from_file(fname, auto_dir=auto_dir)
a2 = PyAuto.from_file(fname2, auto_dir=auto_dir)

############
# plotting #
############

# principle continuation in eta
###############################

fig, axes = plt.subplots(nrows=4, figsize=(6, 6), dpi=dpi)

# plot principle eta continuation for different alphas
n_etas = 4
for i in range(n_etas):
    ax = axes[i]
    if i == 0:
        ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta', ax=ax)
        ax = a2.plot_continuation('PAR(1)', 'U(1)', cont=f'eta', ax=ax, line_color_stable='#8299b0',
                                  line_color_unstable='#8299b0')
    else:
        ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_{i + 1}', ax=ax)
        ax = a2.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_{i + 1}', ax=ax, line_color_stable='#8299b0',
                                  line_color_unstable='#8299b0')
    ax.set_xlabel(r'$\bar\eta$')
    ax.set_ylabel('r')
plt.tight_layout()

# 2D continuations in eta and (alpha, U0)
#########################################

fig2, axes2 = plt.subplots(ncols=2, figsize=(6, 2), dpi=dpi)

ax = axes2[0]
a.plot_continuation('PAR(1)', 'PAR(5)', cont='eta_Delta_lp1', ax=ax, line_style_unstable='solid',
                    line_color_stable='#8299b0', line_color_unstable='#8299b0', ignore=['GH'])
a2.plot_continuation('PAR(1)', 'PAR(7)', cont='eta_Delta_lp1', ax=ax, line_style_stable='dotted',
                     line_style_unstable='dotted', line_color_stable='#8299b0', line_color_unstable='#8299b0')
a.plot_continuation('PAR(1)', 'PAR(5)', cont='eta_Delta_hb1', ax=ax, line_style_unstable='solid', ignore=['GH'])
a2.plot_continuation('PAR(1)', 'PAR(7)', cont='eta_Delta_hb1', ax=ax, line_style_stable='dotted',
                     line_style_unstable='dotted')
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\Delta$')
ax.set_title(r'$\alpha = 0.04$, $U_0 = 1.0$')
ax.set_xlim([-1.0, 1.0])

ax = axes2[1]
a.plot_continuation('PAR(1)', 'PAR(5)', cont='eta_Delta_lp2', ax=ax, line_style_unstable='solid')
a2.plot_continuation('PAR(1)', 'PAR(7)', cont='eta_Delta_lp2', ax=ax,line_color_stable='#8299b0',
                     line_color_unstable='#8299b0')
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\Delta$')
ax.set_title(r'$\alpha = 0.0$, $U_0 = 0.2$')

plt.tight_layout()

plt.show()
