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

c1 = True  # bistable state
c2 = False  # oscillatory state

fname = 'results/stn_gpe_lcs_c1.pkl' if c1 else 'results/stn_gpe_lcs_c2.pkl'
condition = 'c1' if c1 else 'c2'
a = PyAuto.from_file(fname, auto_dir=auto_dir)

# oscillatory regime
####################

fig1 = plt.figure(tight_layout=True, figsize=(6.0, 6.0), dpi=dpi)
grid1 = gs.GridSpec(2, 2)

fig2 = plt.figure(tight_layout=True, figsize=(5.0, 6.0), dpi=dpi)
grid2 = gs.GridSpec(3, 1)

if c2:

    # 2D Plots
    ##########

    # k_gp x k_pe: healthy state
    ax = fig1.add_subplot(grid1[0, 0])
    ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'{condition}:k_pe/k_gp:lp1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'{condition}:k_pe/k_gp:hb1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'{condition}:k_pe/k_gp:hb2', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{pe}$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    # k_ae x k_pe: healthy state
    ax = fig1.add_subplot(grid1[0, 1])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}:k_ae/k_gp:lp1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}:k_ae/k_gp:hb1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}:k_ae/k_gp:hb2', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{ae}$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    # k_ae x k_pe: early PD state
    ax = fig1.add_subplot(grid1[1, 0])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}.2:k_ae/k_gp:hb1', ax=ax,
                             line_color_stable='#3689c9', line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}.2:k_ae/k_gp:hb2', ax=ax,
                             line_color_stable='#148F77', line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{ae}$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    # k_ae x k_pe: late PD state
    ax = fig1.add_subplot(grid1[1, 1])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}.3:k_ae/k_gp:hb1', ax=ax,
                             line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}.3:k_ae/k_gp:hb2', ax=ax,
                             line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}.3:k_ae/k_gp:hb3', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'], line_color_stable='#ee2b2b',
                             line_color_unstable='#ee2b2b')
    ax.set_ylabel(r'$k_{ae}$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    plt.tight_layout()
    plt.savefig(f'stn_gpe_{condition}_2ds.svg')

    # 1D continuations
    ##################

    # healthy state
    ax = fig2.add_subplot(grid2[0, 0])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}:k_gp', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}:k_gp:lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}:k_gp:lc2', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    ax.set_ylabel('$firing rate (GPe-p)$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 20.0])
    # ax.set_ylim([0.0, 20.0])

    # early PD stage
    ax = fig2.add_subplot(grid2[1, 0])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.2:k_gp', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.2:k_gp:lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.2:k_gp:lc2', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    #ax.set_ylabel('$firing rate (GPe-p)$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 20.0])
    # ax.set_ylim([0.0, 20.0])

    # late PD stage
    ax = fig2.add_subplot(grid2[2, 0])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.3:k_gp', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.3:k_gp:lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.3:k_gp:lc2', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.3:k_gp:lc3', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    #ax.set_ylabel('$firing rate (GPe-p)$')
    ax.set_xlabel(r'$k_{pe}$')
    ax.set_xlim([0.0, 20.0])
    # ax.set_ylim([0.0, 20.0])

    plt.tight_layout()
    plt.savefig(f'stn_gpe_{condition}_lcs.svg')

elif c1:

    # 2D Plots
    ##########

    # k_gp x k_pe: healthy state
    ax = fig1.add_subplot(grid1[0, 0])
    ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'{condition}:k_pe/k_gp:lp1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'{condition}:k_pe/k_gp:hb1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'{condition}:k_pe/k_gp:hb2', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{pe}$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    # k_ae x k_pe: healthy state
    ax = fig1.add_subplot(grid1[0, 1])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}:k_ae/k_gp:lp1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}:k_ae/k_gp:lp2', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}:k_ae/k_gp:hb1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}:k_ae/k_gp:hb2', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{ae}$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    # k_ae x k_pe: early PD state
    ax = fig1.add_subplot(grid1[1, 0])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}.2:k_ae/k_gp:lp1', ax=ax,
                             line_color_stable='#3689c9', line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}.2:k_ae/k_gp:hb1', ax=ax,
                             line_color_stable='#2cc7d8', line_color_unstable='#2cc7d8', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}.2:k_ae/k_gp:hb2', ax=ax,
                             line_color_stable='#148F77', line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}.2:k_ae/k_gp:hb3', ax=ax,
                             line_color_stable='#ee2b2b', line_color_unstable='#ee2b2b', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}.2:k_ae/k_gp:hb4', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{ae}$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    # k_ae x k_pe: late PD state
    ax = fig1.add_subplot(grid1[1, 1])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}.3:k_ae/k_gp:lp1', ax=ax,
                             line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(6)', cont=f'{condition}.3:k_ae/k_gp:hb1', ax=ax,
                             line_color_stable='#148F77', line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{ae}$')
    ax.set_xlabel(r'$k_{gp}$')
    # ax.set_xlim([0.0, 20.0])
    # ax.set_ylim([0.0, 20.0])

    #plt.tight_layout()
    plt.savefig(f'stn_gpe_{condition}_fig1.svg')

    # 1D continuations
    ##################

    # healthy state
    ax = fig2.add_subplot(grid2[0, 0])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}:k_gp', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}:k_gp:lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    ax.set_ylabel('$firing rate (GPe-p)$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 20.0])
    # ax.set_ylim([0.0, 20.0])

    # early PD stage
    ax = fig2.add_subplot(grid2[1, 0])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.2:k_gp', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.4:k_gp', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.2:k_gp:lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.2:k_gp:lc2', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.2:k_gp:lc3', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.2:k_gp:lc4', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    # ax.set_ylabel('$firing rate (GPe-p)$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_xlim([0.0, 20.0])
    # ax.set_ylim([0.0, 20.0])

    # late PD stage
    ax = fig2.add_subplot(grid2[2, 0])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.3:k_gp', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1, ignore=['UZ'])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.3:k_gp:lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    # ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.3:k_gp:lc2', ax=ax, line_color_stable='#148F77',
    #                          line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    # ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.3:k_gp:lc3', ax=ax, line_color_stable='#148F77',
    #                          line_color_unstable='#148F77', default_size=markersize1, ignore=['UZ'])
    # ax.set_ylabel('$firing rate (GPe-p)$')
    ax.set_xlabel(r'$k_{pe}$')
    ax.set_xlim([0.0, 20.0])
    # ax.set_ylim([0.0, 20.0])

    plt.tight_layout()
    plt.savefig(f'stn_gpe_{condition}_lcs.svg')

plt.show()
