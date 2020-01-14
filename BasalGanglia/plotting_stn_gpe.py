import matplotlib.pyplot as plt
import matplotlib as mpl
from pyauto import PyAuto

################
# file loading #
################

fname = 'results/qif_stn_gpe.pkl'
a = PyAuto.from_file(fname)

###################################
# plotting of qif_stn_gpe results #
###################################

# continuation in alpha
#######################

fig0, ax0 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# codim 1
a.plot_continuation('PAR(17)', 'U(3)', cont='alpha', ax=ax0[0, 0], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax0[0, 0].set_xlabel('alpha')
ax0[0, 0].set_ylabel('r_i')

# codim 2
a.plot_continuation('PAR(17)', 'PAR(3)', cont='eta_str/alpha', ax=ax0[0, 1], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax0[0, 1].set_xlabel('alpha')
ax0[0, 1].set_ylabel('eta_str')
a.plot_continuation('PAR(17)', 'PAR(9)', cont='k/alpha', ax=ax0[1, 0], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax0[1, 0].set_xlabel('alpha')
ax0[1, 0].set_ylabel('k')
a.plot_continuation('PAR(17)', 'PAR(10)', cont='k_i/alpha', ax=ax0[1, 1], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax0[1, 1].set_xlabel('alpha')
ax0[1, 1].set_ylabel('k_i')

plt.tight_layout()

# continuation in eta_str
#########################

# codim 1
fig1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
a.plot_continuation('PAR(3)', 'U(3)', cont='eta_str', ax=ax1[0, 0], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(3)', 'U(3)', cont=f'eta_str_alpha_lc', ax=ax1[0, 0], ignore=['BP'], line_color_stable='#148F77')
ax1[0, 0].set_xlabel('eta_str')
ax1[0, 0].set_ylabel('r_i')

# codim 2
a.plot_continuation('PAR(3)', 'PAR(7)', cont='k_ie/eta_str', ax=ax1[0, 1], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax1[0, 1].set_xlabel('eta_str')
ax1[0, 1].set_ylabel('k_ie')
a.plot_continuation('PAR(3)', 'PAR(17)', cont='alpha/eta_str', ax=ax1[1, 0], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax1[1, 0].set_xlabel('eta_str')
ax1[1, 0].set_ylabel('alpha')
a.plot_continuation('PAR(3)', 'PAR(17)', cont='eta_str_alpha_lc', ax=ax1[1, 1], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax1[1, 1].set_xlabel('eta_str')
ax1[1, 1].set_ylabel('alpha')

plt.tight_layout()

# continuation in k
###################

# codim 1
fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
a.plot_continuation('PAR(9)', 'U(3)', cont='k', ax=ax2[0, 0], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(9)', 'U(3)', cont=f'k_alpha_lc', ax=ax2[0, 0], ignore=['BP'], line_color_stable='#148F77')
ax2[0, 0].set_xlabel('k')
ax2[0, 0].set_ylabel('r_i')

# codim 2
a.plot_continuation('PAR(9)', 'PAR(10)', cont='k/k_i', ax=ax2[0, 1], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax2[0, 1].set_xlabel('k')
ax2[0, 1].set_ylabel('k_i')
a.plot_continuation('PAR(9)', 'PAR(17)', cont='alpha/k', ax=ax2[1, 0], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax2[1, 0].set_xlabel('k')
ax2[1, 0].set_ylabel('alpha')
a.plot_continuation('PAR(9)', 'PAR(17)', cont='k_alpha_lc', ax=ax2[1, 1], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax2[1, 1].set_xlabel('k')
ax2[1, 1].set_ylabel('alpha')

plt.tight_layout()

# continuation in k_i
#####################

# codim 1
fig3, ax3 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
a.plot_continuation('PAR(10)', 'U(3)', cont='k_i', ax=ax3[0, 0], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
#a.plot_continuation('PAR(10)', 'U(3)', cont=f'k_i_alpha_lc', ax=ax3[0, 0], ignore=['BP'], line_color_stable='#148F77')
ax3[0, 0].set_xlabel('k_i')
ax3[0, 0].set_ylabel('r_i')

# codim 2
a.plot_continuation('PAR(10)', 'PAR(9)', cont='k_i/k', ax=ax3[0, 1], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax3[0, 1].set_xlabel('k_i')
ax3[0, 1].set_ylabel('k')
a.plot_continuation('PAR(10)', 'PAR(17)', cont='alpha/k_i', ax=ax3[1, 0], line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
ax3[1, 0].set_xlabel('k_i')
ax3[1, 0].set_ylabel('alpha')
#a.plot_continuation('PAR(10)', 'PAR(17)', cont='k_i_alpha_lc', ax=ax3[1, 1], line_color_stable='#76448A',
#                    line_color_unstable='#5D6D7E')
#ax3[1, 1].set_xlabel('k_i')
#ax3[1, 1].set_ylabel('alpha')

plt.tight_layout()
plt.show()
