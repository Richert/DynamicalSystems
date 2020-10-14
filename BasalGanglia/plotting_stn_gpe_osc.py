import matplotlib.pyplot as plt
import matplotlib as mpl
from pyrates.utility.pyauto import PyAuto
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

c1 = True  # bistable state
c2 = False  # oscillatory state

fname = 'results/stn_gpe_osc_c1.pkl' if c1 else 'results/stn_gpe_osc_c2.pkl'
condition = 'c1' if c1 else 'c2'
a = PyAuto.from_file(fname, auto_dir='~/PycharmProjects/auto-07p')

# STN - GPe-p oscillations
##########################

fig1 = plt.figure(tight_layout=True, figsize=(6.0, 4.0), dpi=dpi)
grid1 = gs.GridSpec(2, 2)

# codim 1
ax = fig1.add_subplot(grid1[0, :])
ax = a.plot_continuation('PAR(2)', 'U(3)', cont=f'{condition}:eta_p', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(2)', 'U(3)', cont=f'{condition}:eta_p:lc1', ax=ax, line_color_stable='#148F77',
                         line_color_unstable='#148F77', default_size=markersize1)
ax = a.plot_continuation('PAR(2)', 'U(3)', cont=f'{condition}:eta_p:lc2', ax=ax, line_color_stable='#148F77',
                         line_color_unstable='#148F77', default_size=markersize1)
ax.set_xlabel(r'$\eta_{p}$')
ax.set_ylabel('Firing rate (GPe-p)')
if c1:
    pass
else:
    ax.set_xlim([-5.0, 7.0])
    ax.set_ylim([0.0, 0.3])
    ax.set_yticklabels([0.0, 100.0, 200.0, 300.0])

# codim 2
ax = fig1.add_subplot(grid1[1, 0])
ax = a.plot_continuation('PAR(2)', 'PAR(5)', cont=f'{condition}:k_pe/eta_p:hb1', ax=ax, line_color_stable='#8299b0',
                         line_color_unstable='#8299b0', default_size=markersize1,
                         line_style_unstable='solid', ignore=['LP', 'UZ'])
ax = a.plot_continuation('PAR(2)', 'PAR(5)', cont=f'{condition}:k_pe/eta_p:hb2', ax=ax, line_color_stable='#3689c9',
                         line_color_unstable='#3689c9', default_size=markersize1,
                         line_style_unstable='solid', ignore=['LP', 'UZ'])
ax.set_ylabel(r'$k_{pe}$')
ax.set_xlabel(r'$\eta_p$')
if c1:
    pass
else:
    ax.set_ylim([0.0, 5.0])
    ax.set_xlim([-10.0, 10.0])

ax = fig1.add_subplot(grid1[1, 1])
ax = a.plot_continuation('PAR(2)', 'PAR(10)', cont=f'{condition}:k_pa/eta_p:hb1', ax=ax, line_color_stable='#8299b0',
                         line_color_unstable='#8299b0', default_size=markersize1,
                         line_style_unstable='solid', ignore=['LP', 'UZ'])
ax = a.plot_continuation('PAR(2)', 'PAR(10)', cont=f'{condition}:k_pa/eta_p:hb2', ax=ax, line_color_stable='#3689c9',
                         line_color_unstable='#3689c9', default_size=markersize1,
                         line_style_unstable='solid', ignore=['LP', 'UZ'])
ax.set_ylabel(r'$k_{pa}$')
ax.set_xlabel(r'$\eta_p$')
if c1:
    pass
else:
    ax.set_ylim([0.0, 5.0])
    ax.set_xlim([-5.0, 15.0])

plt.tight_layout()
plt.savefig(f'stn_gpe_{condition}_fig1.svg')

# impact of adding GPe-a to GPe description
###########################################

fig2 = plt.figure(tight_layout=True, figsize=(6.0, 4.0), dpi=dpi)
grid2 = gs.GridSpec(2, 2)

# codim 1
ax = fig2.add_subplot(grid2[0, :])
ax = a.plot_continuation('PAR(2)', 'U(3)', cont=f'{condition}.2:eta_p', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E', default_size=markersize1)
ax = a.plot_continuation('PAR(2)', 'U(3)', cont=f'{condition}.2:eta_p:lc1', ax=ax, line_color_stable='#148F77',
                         line_color_unstable='#148F77', default_size=markersize1)
ax = a.plot_continuation('PAR(2)', 'U(3)', cont=f'{condition}.2:eta_p:lc2', ax=ax, line_color_stable='#148F77',
                         line_color_unstable='#148F77', default_size=markersize1)
ax.set_xlabel(r'$\eta_p$')
ax.set_ylabel('Firing rate (GPe-p)')
if c1:
    pass
else:
    ax.set_xlim([-4.0, 8.0])
    ax.set_ylim([0.0, 0.4])
    ax.set_yticklabels([0.0, 100.0, 200.0, 300.0, 400.0])

# codim 2
ax = fig2.add_subplot(grid2[1, 0])
ax = a.plot_continuation('PAR(10)', 'PAR(5)', cont=f'{condition}.2:k_pe/k_pa:hb1', ax=ax, line_color_stable='#8299b0',
                         line_color_unstable='#8299b0', default_size=markersize1,
                         line_style_unstable='solid', ignore=['LP', 'UZ'])
ax = a.plot_continuation('PAR(10)', 'PAR(5)', cont=f'{condition}.2:k_pe/k_pa:hb2', ax=ax, line_color_stable='#3689c9',
                         line_color_unstable='#3689c9', default_size=markersize1,
                         line_style_unstable='solid', ignore=['LP', 'UZ'])
ax.set_ylabel(r'$k_{pe}$')
ax.set_xlabel(r'$k_{pa}$')
if c1:
    pass
else:
    ax.set_ylim([0.0, 10.0])
    ax.set_xlim([-3.0, 3.0])

ax = fig2.add_subplot(grid2[1, 1])
ax = a.plot_continuation('PAR(3)', 'PAR(6)', cont=f'{condition}.2:k_ae/eta_a:hb1', ax=ax, line_color_stable='#8299b0',
                         line_color_unstable='#8299b0', default_size=markersize1,
                         line_style_unstable='solid', ignore=['LP', 'UZ'])
ax = a.plot_continuation('PAR(3)', 'PAR(6)', cont=f'{condition}.2:k_ae/eta_a:hb2', ax=ax, line_color_stable='#3689c9',
                         line_color_unstable='#3689c9', default_size=markersize1,
                         line_style_unstable='solid', ignore=['LP', 'UZ'])
ax.set_ylabel(r'$k_{ae}$')
ax.set_xlabel(r'$\eta_a$')
if c1:
    pass
else:
    ax.set_ylim([0.0, 10.0])
    ax.set_xlim([0.0, 2.5])

plt.tight_layout()
plt.savefig(f'stn_gpe_{condition}_fig2.svg')

# impact of adding STN -> GPe-a projection
##########################################

if c1:

    fig3 = plt.figure(tight_layout=True, figsize=(6.0, 6.0), dpi=dpi)
    grid3 = gs.GridSpec(3, 2)

    # codim 1
    ax = fig3.add_subplot(grid3[0, :])
    ax = a.plot_continuation('PAR(3)', 'U(3)', cont=f'{condition}.3:eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax = a.plot_continuation('PAR(3)', 'U(3)', cont=f'{condition}.3:eta_a:lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1)
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel('Firing rate (GPe-p)')
    # ax.set_xlim([0.0, 5.0])
    # ax.set_ylim([0.02, 0.15])
    # ax.set_yticklabels([0.0, 50.0, 100.0, 150.0])

    ax = fig3.add_subplot(grid3[1, :])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.3:k_gp', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.3:k_gp:lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1)
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_ylabel('Firing rate (GPe-p)')
    # ax.set_xlim([0.0, 5.0])
    # ax.set_ylim([0.02, 0.15])
    # ax.set_yticklabels([0.0, 50.0, 100.0, 150.0])

    # codim 2
    ax = fig3.add_subplot(grid3[2, 0])
    ax = a.plot_continuation('PAR(9)', 'PAR(22)', cont=f'{condition}.3:k_ap/k_gp:lp', ax=ax, line_color_stable='#8299b0',
                             line_color_unstable='#8299b0', default_size=markersize1,
                             line_style_unstable='solid', ignore=['LP', 'UZ'])
    ax = a.plot_continuation('PAR(9)', 'PAR(22)', cont=f'{condition}.3:k_ap/k_gp:hb', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['LP', 'UZ'])
    ax.set_ylabel(r'$k_{ap}$')
    ax.set_xlabel(r'$k_{gp}$')
    # ax.set_ylim([0.0, 2.0])
    # ax.set_xlim([0.0, 6.0])

    ax = fig3.add_subplot(grid3[2, 1])
    ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'{condition}.3:k_ep/k_gp:lp', ax=ax, line_color_stable='#8299b0',
                             line_color_unstable='#8299b0', default_size=markersize1,
                             line_style_unstable='solid', ignore=['LP', 'UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'{condition}.3:k_ep/k_gp:hb', ax=ax,
                             line_color_stable='#3689c9', line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['LP', 'UZ'])
    ax.set_ylabel(r'$k_{ep}$')
    ax.set_xlabel(r'$k_{gp}$')
    # ax.set_ylim([0.0, 2.0])
    # ax.set_xlim([0.0, 6.0])

else:

    fig3 = plt.figure(tight_layout=True, figsize=(6.0, 6.0), dpi=dpi)
    grid3 = gs.GridSpec(3, 2)

    # codim 1
    ax = fig3.add_subplot(grid3[0, :])
    ax = a.plot_continuation('PAR(9)', 'U(3)', cont=f'{condition}.3:k_ap', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax = a.plot_continuation('PAR(9)', 'U(3)', cont=f'{condition}.3:k_ap:lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1)
    ax.set_xlabel(r'$k_{ap}$')
    ax.set_ylabel('Firing rate (GPe-p)')
    ax.set_xlim([0.0, 5.0])
    ax.set_ylim([0.02, 0.15])
    ax.set_yticklabels([0.0, 50.0, 100.0, 150.0])

    ax = fig3.add_subplot(grid3[1, :])
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.3:k_gp', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', default_size=markersize1)
    ax = a.plot_continuation('PAR(22)', 'U(3)', cont=f'{condition}.3:k_gp:lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', default_size=markersize1)
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_ylabel('Firing rate (GPe-p)')
    ax.set_xlim([0.0, 5.0])
    ax.set_ylim([0.02, 0.15])
    ax.set_yticklabels([0.0, 50.0, 100.0, 150.0])

    # codim 2
    ax = fig3.add_subplot(grid3[2, 0])
    ax = a.plot_continuation('PAR(22)', 'PAR(9)', cont=f'{condition}.3:k_gp/k_ap', ax=ax, line_color_stable='#8299b0',
                             line_color_unstable='#8299b0', default_size=markersize1,
                             line_style_unstable='solid', ignore=['LP', 'UZ'])
    ax = a.plot_continuation('PAR(22)', 'PAR(9)', cont=f'{condition}.3:k_ap/k_gp', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', default_size=markersize1,
                             line_style_unstable='solid', ignore=['LP', 'UZ'])
    ax.set_ylabel(r'$k_{ap}$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_ylim([0.0, 2.0])
    ax.set_xlim([0.0, 6.0])

    ax = fig3.add_subplot(grid3[2, 1])
    ax = a.plot_continuation('PAR(22)', 'PAR(5)', cont=f'{condition}.3:k_pe/k_gp', ax=ax, line_color_stable='#8299b0',
                             line_color_unstable='#8299b0', default_size=markersize1,
                             line_style_unstable='solid', ignore=['LP', 'UZ'])
    ax.set_ylabel(r'$k_{pe}$')
    ax.set_xlabel(r'$k_{gp}$')
    ax.set_ylim([0.0, 2.0])
    ax.set_xlim([0.0, 6.0])

plt.tight_layout()
plt.savefig(f'stn_gpe_{condition}_fig3.svg')

plt.show()
