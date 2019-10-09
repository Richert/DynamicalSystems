import matplotlib.pyplot as plt
import matplotlib as mpl
from pyauto import PyAuto

################
# file loading #
################

fname = 'results/eic.pkl'
a = PyAuto.from_file(fname)

############
# plotting #
############

# principle continuation in eta
###############################

fig, axes = plt.subplots(ncols=2)
ax = axes[0]
n = 1
for i in range(n):
    ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_{i}', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E')
ax = axes[1]
ax = a.plot_continuation('PAR(1)', 'U(1)', cont='eta_0', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E')
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_hb1', ax=ax, ignore=['BP'], line_color_stable='#148F77')
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_hb2', ax=ax, ignore=['BP'], line_color_stable='#148F77')

#ax = axes[1, 0]
#ax = a.plot_continuation('PAR(8)', 'U(1)', cont=f'beta_hb', ax=ax, ignore=['BP'], line_color_stable='#148F77')

#ax = axes[1, 1]
#ax = a.plot_continuation('PAR(7)', 'U(1)', cont=f'alpha_hb', ax=ax, ignore=['BP'], line_color_stable='#148F77')

plt.tight_layout()

# limit cycle continuation in beta
##################################

fig2, axes2 = plt.subplots(nrows=2)

ax = axes2[0]
ax = a.plot_continuation('PAR(8)', 'U(1)', cont='beta_hb1', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E')
ax = axes2[1]
ax = a.plot_timeseries('U(1)', cont='beta_hb1', points=['LP1', 'TR1'], ax=ax,
                       linespecs=[{'colors': '#76448A'}, {'colors': '#5D6D7E'}])


#axes2 = a.plot_continuation('PAR(8)', 'U(1)', cont='beta_bp2', ax=axes2, line_color_stable='#76448A',
#                            line_color_unstable='#5D6D7E')
# codim 2 bifurcations
# fig2, axes2 = plt.subplots(nrows=2)
#
# # plot eta-beta continuation of the limit cycle
# ax = axes2[0]
# ax = a.plot_continuation('PAR(7)', 'PAR(9)', cont=f'taua_beta_hb1', ax=ax, ignore=['LP', 'BP'],
#                          line_style_unstable='solid')
# ax = a.plot_continuation('PAR(7)', 'PAR(9)', cont=f'taua_beta_hb2', ax=ax, ignore=['LP', 'BP'],
#                          line_style_unstable='solid')
#
# ax = axes2[1]
# ax = a.plot_continuation('PAR(2)', 'PAR(9)', cont=f'etai_beta_hb1', ax=ax, ignore=['LP', 'BP'],
#                          line_style_unstable='solid')
# ax = a.plot_continuation('PAR(2)', 'PAR(9)', cont=f'etai_beta_hb2', ax=ax, ignore=['LP', 'BP'],
#                          line_style_unstable='solid')

plt.show()
