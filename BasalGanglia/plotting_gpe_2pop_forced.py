import matplotlib.pyplot as plt
import matplotlib as mpl
from pyauto import PyAuto
from pyrates.utility.visualization import plot_connectivity
import matplotlib.gridspec as gs
import numpy as np

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

c1 = True  # oscillatory
c2 = False  # bistable

if c1:

    fname = 'results/gpe_2pop_forced_lc.pkl'
    a = PyAuto.from_file(fname)

    # continuation of alpha and omega
    #################################

    fig1 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
    grid1 = gs.GridSpec(3, 1)

    # continuation of the limit cycle in alpha
    ax = fig1.add_subplot(grid1[0, :])
    ax = a.plot_continuation('PAR(23)', 'U(2)', cont='c1:alpha', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1, ignore=['UZ'],
                             custom_bf_styles={'TR': {'marker': 'X'}})
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Firing rate')
    ax.set_yticks([0.04, 0.08, 0.12])
    ax.set_yticklabels([40.0, 80.0, 120.0])

    # continuation of the torus bifurcation in alpha and omega
    ax = fig1.add_subplot(grid1[1:, :])
    i = 1
    while i < 50:
        try:
            ax = a.plot_continuation('PAR(23)', 'PAR(25)', cont=f'c1:omega/alpha/TR{i}', ax=ax, ignore=['UZ', 'R1'],
                                     line_color_stable='#148F77', line_color_unstable='#148F77',
                                     custom_bf_styles={'R1': {'marker': 'h', 'color': 'k'},
                                                       'R2': {'marker': 'h', 'color': 'g'},
                                                       'R3': {'marker': 'h', 'color': 'r'},
                                                       'R4': {'marker': 'h', 'color': 'b'}},
                                     line_style_unstable='solid', default_size=markersize1)
            i += 1
            j = 1
            while j < 10:
                try:
                    ax = a.plot_continuation('PAR(23)', 'PAR(25)', cont=f'c1:alpha/omega/TR{i}/R4{j}', ax=ax,
                                             ignore=['UZ'], line_color_stable='#b8b632', line_color_unstable='#b8b632',
                                             line_style_unstable='solid', default_size=markersize1)
                    j += 1
                except KeyError:
                    break
            j = 1
            while j < 10:
                try:
                    ax = a.plot_continuation('PAR(23)', 'PAR(25)', cont=f'c1:alpha/omega/TR{i}/R3{j}', ax=ax,
                                             ignore=['UZ'], line_color_stable='#8299b0', line_color_unstable='#8299b0',
                                             line_style_unstable='solid', default_size=markersize1)
                    j += 1
                except KeyError:
                    break
        except KeyError:
            break
    i = 1
    while i < 20:
        try:
            ax = a.plot_continuation('PAR(23)', 'PAR(25)', cont=f'c1:omega/alpha/PD{i}', ax=ax, ignore=['UZ'],
                                     line_color_stable='#3689c9', line_color_unstable='#3689c9',
                                     line_style_unstable='solid', default_size=markersize1)
            i += 1
            j = 1
            while j < 10:
                try:
                    ax = a.plot_continuation('PAR(23)', 'PAR(25)', cont=f'c1:alpha/omega/PD{i}/R4{j}', ax=ax,
                                             ignore=['UZ'], line_color_stable='#b8b632', line_color_unstable='#b8b632',
                                             line_style_unstable='solid', default_size=markersize1)
                    j += 1
                except KeyError:
                    break
            j = 1
            while j < 10:
                try:
                    ax = a.plot_continuation('PAR(23)', 'PAR(25)', cont=f'c1:alpha/omega/PD{i}/R3{j}', ax=ax,
                                             ignore=['UZ'], line_color_stable='#8299b0', line_color_unstable='#8299b0',
                                             line_style_unstable='solid', default_size=markersize1)
                    j += 1
                except KeyError:
                    break
        except KeyError:
            break

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\omega$')
    fig1.canvas.draw()
    #ax.set_yticklabels([label._y + 5.0 for label in ax.get_yticklabels()])
    #ax.set_xlim([0.0, 40.0])
    #ax.set_ylim([55.0, 85.0])

    plt.tight_layout()
    plt.show()

    # lyapunov exponents and fractal dimension
    ##########################################

    # fig2 = plt.figure(tight_layout=True, figsize=(6.0, 4.0), dpi=dpi)
    # grid2 = gs.GridSpec(1, 2)
    #
    # # maximal lyapunov exponent
    # ax = fig2.add_subplot(grid2[0, 0])
    # ax = plot_connectivity(a.LE_max.T, ax=ax, threshold=False)
    # ax.invert_yaxis()
    # ax.set_xticks(ax.get_xticks()[0::4])
    # ax.set_yticks(ax.get_yticks()[0::4])
    # ax.set_xticklabels(np.round(a.omega[0::4], decimals=1))
    # ax.set_yticklabels(np.round(a.alpha[0::4], decimals=1))
    #
    # # fractal dimension
    # ax = fig2.add_subplot(grid2[0, 1])
    # ax = plot_connectivity(a.D_ky.T, ax=ax)
    # ax.invert_yaxis()
    # ax.set_xticks(ax.get_xticks()[0::4])
    # ax.set_yticks(ax.get_yticks()[0::4])
    # ax.set_xticklabels(np.round(a.omega[0::4], decimals=1))
    # ax.set_yticklabels(np.round(a.alpha[0::4], decimals=1))

    plt.show()

if c2:

    fname = 'results/gpe_2pop_forced_bs.pkl'
    a = PyAuto.from_file(fname)

    # continuation of alpha and omega
    #################################

    fig2 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
    grid2 = gs.GridSpec(3, 1)

    # continuation of the limit cycle in alpha
    ax = fig2.add_subplot(grid2[0, :])
    ax = a.plot_continuation('PAR(23)', 'U(2)', cont='c2:alpha', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1, ignore=['UZ'])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Firing rate')
    ax.set_yticks([0.04, 0.05, 0.06, 0.07])
    ax.set_yticklabels([40.0, 50.0, 60.0, 70.0])

    # continuation of the torus bifurcation in alpha and omega
    ax = fig2.add_subplot(grid2[1:, :])
    ax = a.plot_continuation('PAR(23)', 'PAR(25)', cont='c2:alpha/omega', ax=ax, ignore=['UZ'],
                             line_color_stable='#148F77', default_size=markersize1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\omega$')
    #ax.set_ylim([50.0, 100.0])
    fig2.canvas.draw()
    ax.set_yticklabels([label._y + 5.0 for label in ax.get_yticklabels()])
    #ax.set_xlim([0.0, 40.0])

    plt.tight_layout()
    plt.show()
