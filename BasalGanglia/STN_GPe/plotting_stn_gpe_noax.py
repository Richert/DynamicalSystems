import matplotlib.pyplot as plt
import matplotlib as mpl
from pyrates.utility.pyauto import PyAuto
import matplotlib.gridspec as gs
import sys

# plotting parameters
# linewidth = 2.0
# fontsize1 = 12
# fontsize2 = 14
# markersize1 = 80
# markersize2 = 80
linewidth = 1.2
fontsize1 = 10
fontsize2 = 12
markersize1 = 60
markersize2 = 60
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


############################################
# file loading and condition specification #
############################################

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

c1 = False  # bistable state
c2 = True  # oscillatory state

model = 'stn_gpe_noax'
condition = 'c1' if c1 else 'c2'
fname = f'results/{model}_{condition}.pkl'
a = PyAuto.from_file(fname, auto_dir=auto_dir)

# oscillatory regime
####################

fig1 = plt.figure(tight_layout=True, figsize=(10.0, 6.0), dpi=dpi)
grid1 = gs.GridSpec(3, 5)


if c2:

    # 2d: k_stn x k_gp
    ax = fig1.add_subplot(grid1[:, :2])
    ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'{condition}:k_stn/k_gp:lp1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ', 'GH'])
    ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'{condition}:k_stn/k_gp:hb1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ', 'GH'])
    ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'{condition}:k_stn/k_gp:hb2', ax=ax,
                             line_color_stable='#ee2b2b',
                             line_color_unstable='#ee2b2b', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ', 'GH'])
    ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'{condition}:k_stn/k_gp:hb3', ax=ax,
                             line_color_stable='#2cc7d8',
                             line_color_unstable='#2cc7d8', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ', 'GH'])
    ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'{condition}:k_stn/k_gp:hb6', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ', 'GH'])
    ax.set_ylabel(r'$k_{stn}$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 25.0])
    ax.set_ylim([0.0, 4.8])

    # 1D: k_stn continuation for k_gp = 3.0
    ax = fig1.add_subplot(grid1[0, 2:])
    ax = a.plot_continuation('PAR(25)', 'U(3)', cont=f'{condition}:k_stn', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(25)', 'U(3)', cont=f'{condition}:k_stn:lc1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ', 'BP'])
    ax.set_xlabel(r'$k_{stn}$')
    #ax.set_ylabel(r'Firing rate (GPe-p)')
    # ax.set_xlim([0.3, 2.0])
    # ax.set_ylim([0.0, 0.2])
    # ax.set_yticks([0.0, 0.1, 0.2])
    # ax.set_yticklabels([0, 100, 200])

    # 1D: k_gp continuation for k_stn = 1.5
    ax = fig1.add_subplot(grid1[1, 2:])
    ax = a.plot_continuation('PAR(25)', 'U(3)', cont=f'{condition}:k_stn:2', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(25)', 'U(3)', cont=f'{condition}:k_stn:2:lc1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ', 'BP'])
    ax.set_xlabel(r'$k_{stn}$')
    #ax.set_ylabel(r'Firing rate (GPe-p)')
    # ax.set_xlim([2.0, 11.0])
    # ax.set_ylim([0.0, 0.4])
    # ax.set_yticks([0.0, 0.2, 0.4])
    # ax.set_yticklabels([0, 200, 400])

    # 1D: k_gp continuation for k_stn = 3.0
    ax = fig1.add_subplot(grid1[2, 2:])
    ax = a.plot_continuation('PAR(25)', 'U(3)', cont=f'{condition}:k_stn:3', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(25)', 'U(3)', cont=f'{condition}:k_stn:3:lc1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ', 'BP'])
    ax = a.plot_continuation('PAR(25)', 'U(3)', cont=f'{condition}:k_stn:3:lc3', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ', 'BP'])
    ax = a.plot_continuation('PAR(25)', 'U(3)', cont=f'{condition}:k_stn:3:lc4', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ', 'BP'])
    ax.set_xlabel(r'$k_{stn}$')
    # ax.set_xlim([4.0, 16.0])
    # ax.set_ylim([0.0, 0.26])
    # ax.set_yticks([0.0, 0.1, 0.2])
    # ax.set_yticklabels([0, 100, 200])

    plt.tight_layout()
    plt.savefig(f'stn_gpe_fre_{condition}_bf.svg')

elif c1:

    # 2D Plots
    ##########

    # 2D Plot
    #########

    # k_stn x k_gp
    ax = fig1.add_subplot(grid1[:, :2])
    ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'{condition}:k_stn/k_gp:lp1', ax=ax,
                             line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'{condition}:k_stn/k_gp:hb1', ax=ax,
                             line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'{condition}:k_stn/k_gp:hb2', ax=ax,
                             line_color_stable='#ee2b2b',
                             line_color_unstable='#ee2b2b', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'{condition}:k_stn/k_gp:hb3', ax=ax,
                             default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'], line_color_stable='#2cc7d8',
                             line_color_unstable='#2cc7d8')
    ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'{condition}:k_gp', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'{condition}:k_stn', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(25)', cont=f'{condition}:k_stn:4', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{stn}$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 25.0])
    ax.set_ylim([0.0, 4.8])

    # 1D: k_stn for k_gp = 3.0
    ax = fig1.add_subplot(grid1[0, 2:])
    ax = a.plot_continuation('PAR(25)', 'U(2)', cont=f'{condition}:k_stn:1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(25)', 'U(2)', cont=f'{condition}:k_stn:1:lc1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ', 'BP'])
    ax = a.plot_continuation('PAR(25)', 'U(2)', cont=f'{condition}:k_stn:1:lc2', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ', 'BP'])
    ax.set_xlabel(r'$k_{stn}$')
    # ax.set_ylabel(r'Firing rate (GPe-p)')
    # ax.set_xlim([0.5, 4.0])
    # ax.set_ylim([0.0, 0.15])
    # ax.set_yticks([0.0, 0.05, 0.1, 0.15])
    # ax.set_yticklabels([0, 50, 100, 150])

    # 1D: k_stn for k_gp = 5.5
    ax = fig1.add_subplot(grid1[1, 2:])
    ax = a.plot_continuation('PAR(25)', 'U(2)', cont=f'{condition}:k_stn:2', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(25)', 'U(2)', cont=f'{condition}:k_stn:2:lc1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ', 'BP'])
    ax = a.plot_continuation('PAR(25)', 'U(2)', cont=f'{condition}:k_stn:2:lc2', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ', 'BP'])
    ax.set_xlabel(r'$k_{stn}$')
    # ax.set_ylabel(r'Firing rate (GPe-p)')
    # ax.set_xlim([0.75, 1.5])
    # ax.set_ylim([0.0, 0.1])
    # ax.set_yticks([0.0, 0.05, 0.1])
    # ax.set_yticklabels([0, 50, 100])

    # 1D: k_stn for k_gp = 10.0
    ax = fig1.add_subplot(grid1[2, 2:])
    ax = a.plot_continuation('PAR(25)', 'U(2)', cont=f'{condition}:k_stn:3', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(25)', 'U(2)', cont=f'{condition}:k_stn:3:lc1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ', 'BP'])
    ax = a.plot_continuation('PAR(25)', 'U(2)', cont=f'{condition}:k_stn:3:lc2', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ', 'BP'])
    ax.set_xlabel(r'$k_{stn}$')
    # ax.set_ylabel(r'Firing rate (GPe-p)')
    # ax.set_xlim([1.5, 5.0])
    # ax.set_ylim([0.0, 0.32])
    # ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    # ax.set_yticklabels([0, 100, 200, 300])

    plt.tight_layout()
    plt.savefig(f'stn_gpe_{condition}_bf.svg')

plt.show()
