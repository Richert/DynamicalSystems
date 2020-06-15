import matplotlib.pyplot as plt
import matplotlib as mpl
from pyauto import PyAuto
import matplotlib.gridspec as gs

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
markersize2 = 40
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

fname = 'results/gpe_2pop_forced.pkl'
a = PyAuto.from_file(fname)

c1 = False  # oscillatory
c2 = True  # bistable

if c1:

    # continuation of alpha and omega
    #################################

    fig1 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
    grid1 = gs.GridSpec(3, 1)

    # continuation of the limit cycle in alpha
    ax = fig1.add_subplot(grid1[0, :])
    ax = a.plot_continuation('PAR(23)', 'U(2)', cont='c1:alpha', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1, ignore=['UZ'],
                             custom_bf_styles={'TR': {'marker': 'X'}})

    # continuation of the torus bifurcation in alpha and omega
    ax = fig1.add_subplot(grid1[1:, :])
    ax = a.plot_continuation('PAR(23)', 'PAR(25)', cont='c1:alpha/omega', ax=ax, ignore=['UZ'],
                             line_color_stable='#148F77', default_size=markersize1,
                             custom_bf_styles={'R1': {'marker': 'h', 'color': 'k'},
                                               'R2': {'marker': 'h', 'color': 'g'},
                                               'R3': {'marker': 'h', 'color': 'r'},
                                               'R4': {'marker': 'h', 'color': 'b'}})

    plt.tight_layout()
    plt.show()

if c2:

    # continuation of alpha and omega
    #################################

    fig2 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
    grid2 = gs.GridSpec(3, 1)

    # continuation of the limit cycle in alpha
    ax = fig2.add_subplot(grid2[0, :])
    ax = a.plot_continuation('PAR(23)', 'U(2)', cont='c2:alpha', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1, ignore=['UZ'])

    # continuation of the torus bifurcation in alpha and omega
    ax = fig2.add_subplot(grid2[1:, :])
    ax = a.plot_continuation('PAR(23)', 'PAR(25)', cont='c2:alpha/omega', ax=ax, ignore=['UZ'],
                             line_color_stable='#148F77', default_size=markersize1)

    plt.tight_layout()
    plt.show()
