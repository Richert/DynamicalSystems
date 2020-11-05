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

fname = 'results/stn_gpe_osc_c1.pkl' if c1 else 'results/stn_gpe_osc_c2.pkl'
condition = 'c1' if c1 else 'c2'
a = PyAuto.from_file(fname, auto_dir=auto_dir)

# oscillatory regime
####################

fig1 = plt.figure(tight_layout=True, figsize=(9.0, 3.0), dpi=dpi)
grid1 = gs.GridSpec(1, 3)

if c2:

    # healthy condition
    ax = fig1.add_subplot(grid1[0, 0])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}:k_pe/k_ae:lp1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}:k_pe/k_ae:hb1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}:k_pe/k_ae:hb2', ax=ax, line_color_stable='#ee2b2b',
                             line_color_unstable='#ee2b2b', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}:k_ae', ax=ax)
    ax.set_ylabel(r'$k_{ae}$')
    ax.set_xlabel(r'$k_{pe}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    # early PD stage
    ax = fig1.add_subplot(grid1[0, 1])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.2:k_pe/k_ae:lp1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.2:k_pe/k_ae:hb1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.2:k_pe/k_ae:hb2', ax=ax, line_color_stable='#ee2b2b',
                             line_color_unstable='#ee2b2b', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.2:k_pe/k_ae:hb3', ax=ax,
                             line_color_stable='#ea2ecb', line_color_unstable='#ea2ecb', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.2:k_pe/k_ae:hb4', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'], line_color_stable='#2cc7d8')
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.2:k_pe/k_ae:hb5', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.2:k_ae', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{ae}$')
    ax.set_xlabel(r'$k_{pe}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    # advanced PD stage
    ax = fig1.add_subplot(grid1[0, 2])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_pe/k_ae:lp1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_pe/k_ae:hb1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_pe/k_ae:hb2', ax=ax, line_color_stable='#ee2b2b',
                             line_color_unstable='#ee2b2b', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_pe/k_ae:hb3', ax=ax,
                             line_color_stable='#ea2ecb', line_color_unstable='#ea2ecb', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_pe/k_ae:hb4', ax=ax,
                             line_color_stable='#2cc7d8', line_color_unstable='#2cc7d8', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_pe/k_ae:hb5', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_ae', ax=ax, default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{ae}$')
    ax.set_xlabel(r'$k_{pe}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    plt.tight_layout()
    plt.savefig(f'stn_gpe_{condition}_fig1.svg')

elif c1:

    # healthy condition
    ax = fig1.add_subplot(grid1[0, 0])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}:k_pe/k_ae:lp1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}:k_pe/k_ae:hb1', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}:k_pe/k_ae:hb2', ax=ax, line_color_stable='#ee2b2b',
                             line_color_unstable='#ee2b2b', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}:k_pe/k_ae:hb3', ax=ax, line_color_stable='#ea2ecb',
                             line_color_unstable='#ea2ecb', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{ae}$')
    ax.set_xlabel(r'$k_{pe}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    # early PD stage
    ax = fig1.add_subplot(grid1[0, 1])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.2:k_pe/k_ae:lp1', ax=ax,
                             line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.2:k_pe/k_ae:hb1', ax=ax,
                             line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.2:k_pe/k_ae:hb2', ax=ax,
                             line_color_stable='#ee2b2b',
                             line_color_unstable='#ee2b2b', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.2:k_pe/k_ae:hb3', ax=ax,
                             line_color_stable='#ea2ecb', line_color_unstable='#ea2ecb', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{ae}$')
    ax.set_xlabel(r'$k_{pe}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    # advanced PD stage
    ax = fig1.add_subplot(grid1[0, 2])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_pe/k_ae:lp1', ax=ax,
                             line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_pe/k_ae:hb1', ax=ax,
                             line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_pe/k_ae:hb2', ax=ax,
                             line_color_stable='#ee2b2b',
                             line_color_unstable='#ee2b2b', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_pe/k_ae:hb3', ax=ax,
                             line_color_stable='#ea2ecb', line_color_unstable='#ea2ecb', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_pe/k_ae:hb4', ax=ax,
                             line_color_stable='#2cc7d8', line_color_unstable='#2cc7d8', default_size=markersize1,
                             line_style_unstable='solid', ignore=['UZ'])
    ax = a.plot_continuation('PAR(5)', 'PAR(6)', cont=f'{condition}.3:k_pe/k_ae:hb5', ax=ax,
                             default_size=markersize1, line_style_unstable='solid', ignore=['UZ'])
    ax.set_ylabel(r'$k_{ae}$')
    ax.set_xlabel(r'$k_{pe}$')
    ax.set_xlim([0.0, 20.0])
    ax.set_ylim([0.0, 20.0])

    plt.tight_layout()
    plt.savefig(f'stn_gpe_{condition}_fig1.svg')

plt.show()
