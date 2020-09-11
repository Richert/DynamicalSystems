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

c1 = [  # bistable state
      False,  # STN -> GPe-p < STN -> GPe-a
      False,   # STN -> GPe-p > STN -> GPe-a
]
c2 = [  # oscillatory state
    False,  # STN -> GPe-p < STN -> GPe-a
    True,  # STN -> GPe-p > STN -> GPe-a
]

if any(c1):
    if c1[0]:
        fname = 'results/stn_gpe_final_c11.pkl'
        condition = 'c1.1'
    else:
        fname = 'results/stn_gpe_final_c12.pkl'
        condition = 'c1.2'
elif any(c2):
    if c2[0]:
        fname = 'results/stn_gpe_final_c21pkl'
        condition = 'c2.1'
    else:
        fname = 'results/stn_gpe_final.pkl'
        condition = 'c2.2'
else:
    fname = 'results/stn_gpe_final.pkl'
    condition = ''

a = PyAuto.from_file(fname)

# continuation of k_gp
######################

fig1 = plt.figure(tight_layout=True, figsize=(6.0, 9.0), dpi=dpi)
grid1 = gs.GridSpec(2, 2)

# codim 1: eta_e
ax = fig1.add_subplot(grid1[:, :])
ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}:k_gp', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)

# codim 1: k_ep
try:
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}:k_gp/lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1)
except KeyError:
    pass
ax.set_xlabel(r'$k_{gp}$')
ax.set_ylabel('Firing rate')

# 2D continuation of Hopf curve
###############################

branch_color = {'HB': '#8299b0',
                'LP': '#b8b632'}

# 2D continuation k_gp x k_gpe_e
################################

if hasattr(a, 'additional_attributes'):
    try:
        fig2 = plt.figure(tight_layout=True, figsize=(6.0, 6.0), dpi=dpi)
        grid2 = gs.GridSpec(2, 2)
        ax = fig2.add_subplot(grid2[:, :])
        for cont in a.additional_attributes['k_gp/k_gp_e:names']:
            if "LP" in cont:
                color = '#8299b0'
            else:
                color = '#3689c9'
            ax = a.plot_continuation('PAR(22)', 'PAR(26)', cont=cont, ax=ax, line_color_stable=color,
                                     line_color_unstable=color, default_size=markersize1,
                                     line_style_unstable='solid', ignore=['LP'])
        ax.set_xlabel(r'$k_{gp}$')
        ax.set_ylabel(r'$k_{stn}$')
        #ax.set_xlim([0.0, 20.0])
        #ax.set_ylim([0.75, 2.0])
        plt.tight_layout()
    except KeyError:
        pass

plt.show()
