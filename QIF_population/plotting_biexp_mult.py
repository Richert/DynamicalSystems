import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#from pyauto import PyAuto
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

fname = 'results/biexp_mult.hdf5'
a = PyAuto.from_file(fname)
#period_solutions = np.fromfile(f"{fname}_period")


############
# plotting #
############

# principle continuation in eta
###############################
'''
fig, axes = plt.subplots(ncols=2, figsize=(7, 1.8), dpi=dpi)

# plot principle eta continuation for different alphas
ax = axes[0]
n_alphas = 6 #5
for i in range(n_alphas):
    try:
        ax=a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_{i}', ax=ax)
    except KeyError:
        pass
ax.set_xlabel(r'$\eta$')

# plot eta continuation for single alpha with limit cycle continuation
ax = axes[1]
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_3', ax=ax)
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta/hb2', ax=ax, ignore=['UZ', 'BP'])
ax.set_xlim((-6.5, -4))
ax.set_xlabel(r'$\eta$')

plt.tight_layout()
plt.show()
'''


cutoff=50
top=400

fig, axes = plt.subplots(3)


fname = 'results/biexp_mult_time_steady.pkl'
b = PyAuto.from_file(fname)
ax = axes[0]
ax = b.plot_continuation('PAR(14)', 'U(1)', cont='steady', ax=ax)
ax.set_xlim((cutoff, top)) # top limit from above
ax.set_xlabel('time')
ax.set_ylabel('firing rate (r)')
ax.set_title(r'Steady state, $\eta=-4.0, \tau_{r}=10.0, \tau_{d}=10.0$')
ax.set_ylim(0,2.5)


fname = 'results/biexp_mult_time_burst.pkl'
b = PyAuto.from_file(fname)
ax = axes[1]
ax = b.plot_continuation('PAR(14)', 'U(1)', cont='burst', ax=ax)
ax.set_xlim((cutoff, top))
ax.set_xlabel('time')
ax.set_ylabel('firing rate (r)')
ax.set_title(r'Oscillatory bursting, $\eta=-5.02, \tau_{r}=10.0, \tau_{d}=10.0$')
ax.set_ylim(0,2.5)


fname = 'results/biexp_mult_time_chaos.pkl'
b = PyAuto.from_file(fname)
ax = axes[2]
ax = b.plot_continuation('PAR(14)', 'U(1)', cont='chaos', ax=ax)
ax.set_xlim((cutoff, top)) # top limit from above
ax.set_xlabel('time')
ax.set_ylabel('firing rate (r)')
ax.set_title(r'Chaotic, $\eta=-5.22, \tau_{r}=0.01, \tau_{d}=10.0$')
ax.set_ylim(0,2.5)

plt.tight_layout()
imagepath='../../plots/'+f'bif_exp_mult_time_time_series.pdf'
plt.savefig(imagepath, format='pdf')

plt.show()




############# Codimension 1 bifurcation diagrams
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 2)

# plot eta continuation
#######################
f_ax1 = fig.add_subplot(gs[0, 0])

#fig, axes = plt.subplots(ncols=2, figsize=(6, 1.8), dpi=dpi)
#ax = axes[0]
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_3', ax=f_ax1)
a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta/hb2', ax=f_ax1,  ignore=['UZ', 'BP'])
f_ax1.set_title(r'1D Continuation in $\bar\eta$ with constant $\tau_r=10$')
f_ax1.set_xlim((-6.5, -4))
f_ax1.axvline(-5.2,0,1, color='blue')
f_ax1.set_xlabel(r'$\bar\eta$')
f_ax1.set_ylabel('Firing rate (r)')
f_ax1.set_ylim(0,2.5)


f_ax2 = fig.add_subplot(gs[0, 1])
#ax = axes[1]
#a.plot_continuation('PAR(4)', 'U(1)', cont=f'tau_r/hb2', ax=ax)
a.plot_continuation('PAR(4)', 'U(1)', cont=f'tau_r/lc', ax=f_ax2)
f_ax2.set_title(r'1D Continuation in $\tau_r$ with constant $\bar\eta=5.2$')
f_ax2.set_xlabel(r'$\tau_r$')
f_ax2.axvline(10,0,1, color='blue')
f_ax2.set_xscale('log')
f_ax2.set_xticks([10,1,0.1])
f_ax2.set_xticklabels([10,1,0.1], fontsize=fontsize1, fontweight='bold')
f_ax2.set_xlim(11, 0.05)  
f_ax2.set_ylim(0,2.5)
f_ax2.set_ylabel('Firing rate (r)')
#plt.tight_layout()


shared_top_y = 5
shared_bottom_y = -0.05

f_ax3 = fig.add_subplot(gs[1, 0])

a.plot_continuation('PAR(1)', 'PAR(4)', cont='tau_r/eta/hb2', ax=f_ax3, ignore=['LP', 'BP', 'UZ'],
                         line_style_unstable='solid', default_size=markersize1)
a.plot_continuation('PAR(1)', 'PAR(4)', cont='tau_r/eta/gh1', ax=f_ax3, ignore=['BP', 'UZ'],
                         line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E', line_style_unstable='dotted',
                         default_size=markersize1)
a.plot_continuation('PAR(1)', 'PAR(4)', cont='tau_r/eta/lc_pd1', ax=f_ax3, ignore=['BP', 'UZ'],
                         line_color_stable='#148F77', line_color_unstable='#148F77', line_style_unstable='dotted',
                         default_size=markersize1)




f_ax3.set_xlabel(r'$\bar\eta$')
f_ax3.set_ylabel(r'$\tau_r$')
# ax.set_xlim([-6.5, -2.5])
f_ax3.set_ylim([shared_bottom_y, shared_top_y])
f_ax3.set_title(r'2D Limit Cycle Continuation in $\tau_r and \bar\eta$')


f_ax4 = fig.add_subplot(gs[1, 1])


a.plot_continuation('PAR(3)', 'PAR(4)', cont='tau_r/alpha/hb2', ax=f_ax4, ignore=['LP', 'BP', 'UZ'],
                         line_style_unstable='solid', default_size=markersize1)
a.plot_continuation('PAR(3)', 'PAR(4)', cont='tau_r/alpha/gh1', ax=f_ax4, ignore=['LP', 'BP', 'UZ'],
                         line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E', line_style_unstable='dotted',
                         default_size=markersize1)
a.plot_continuation('PAR(3)', 'PAR(4)', cont='tau_r/alpha/lc_pd1', ax=f_ax4, ignore=['LP', 'BP', 'UZ', 'R1', 'R2'],
                         line_color_stable='#148F77', line_color_unstable='#148F77', line_style_unstable='dotted',
                         default_size=markersize1)
#ax.set_ylim((-15,15))

f_ax4.set_xlabel(r'$\alpha$')
f_ax4.set_ylabel(r'$\tau$')
# ax.set_xlim([-6.5, -2.5])
f_ax4.set_ylim([shared_bottom_y, shared_top_y])
f_ax4.set_title(r'2D Limit Cycle Continuation in $\tau_r and \alpha$')




imagepath='../../plots/'+f'bif_biexp_mult_eta_tau'+'.pdf'
plt.savefig(imagepath)

plt.show()



'''
############# Codimension 2 bifurcation diagrams

# tau and eta

fig2, ax = plt.subplots(ncols=1, figsize=(3.5, 1.8), dpi=dpi)

# plot eta-tau continuation of the limit cycle

ax = a.plot_continuation('PAR(1)', 'PAR(4)', cont='tau_r/eta/hb2', ax=ax, ignore=['LP', 'BP', 'UZ'],
                         line_style_unstable='solid', default_size=markersize1)
ax = a.plot_continuation('PAR(1)', 'PAR(4)', cont='tau_r/eta/gh1', ax=ax, ignore=['BP', 'UZ'],
                         line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E', line_style_unstable='dotted',
                         default_size=markersize1)
ax = a.plot_continuation('PAR(1)', 'PAR(4)', cont='tau_r/eta/lc_pd1', ax=ax, ignore=['BP', 'UZ'],
                         line_color_stable='#148F77', line_color_unstable='#148F77', line_style_unstable='dotted',
                         default_size=markersize1)
# ax = a.plot_continuation('PAR(5)', 'PAR(4)', cont='tau_r/tau_d/lc_lp1', ax=ax, ignore=['BP', 'UZ'],
#                          line_color_stable='#76448A', line_color_unstable='#76448A', line_style_unstable='dotted',
#                          default_size=markersize1)
#ax.set_ylim((-15,15))

# cosmetics
ax.set_xlabel(r'$\bar\eta$')
ax.set_ylabel(r'$\tau$')
# ax.set_xticks(xticklabels[::3])
# ax.set_yticks(yticklabels[::3])
# ax.set_xlim([-6.5, -2.5])
# ax.set_ylim([0., 0.2])
ax.set_title('2D Limit Cycle Continuation')
plt.tight_layout()
plt.savefig('fig2D_tau_eta.svg')


############# Codimension 2 bifurcation diagrams

# tau and alpha

fig3, ax = plt.subplots(figsize=(3.5, 1.8), dpi=dpi, ncols=1)

# plot eta-tau continuation of the limit cycle

ax = a.plot_continuation('PAR(3)', 'PAR(4)', cont='tau_r/alpha/hb2', ax=ax, ignore=['LP', 'BP', 'UZ'],
                         line_style_unstable='solid', default_size=markersize1)
ax = a.plot_continuation('PAR(3)', 'PAR(4)', cont='tau_r/alpha/gh1', ax=ax, ignore=['LP', 'BP', 'UZ'],
                         line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E', line_style_unstable='dotted',
                         default_size=markersize1)
ax = a.plot_continuation('PAR(3)', 'PAR(4)', cont='tau_r/alpha/lc_pd1', ax=ax, ignore=['LP', 'BP', 'UZ', 'R1', 'R2'],
                         line_color_stable='#148F77', line_color_unstable='#148F77', line_style_unstable='dotted',
                         default_size=markersize1)
#ax.set_ylim((-15,15))


# cosmetics
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\tau$')
# ax.set_xticks(xticklabels[::3])
# ax.set_yticks(yticklabels[::3])
# ax.set_xlim([-6.5, -2.5])
# ax.set_ylim([0., 0.2])
ax.set_title('2D Limit Cycle Continuation')
plt.tight_layout()
#plt.savefig('fig2.svg')

plt.show()
'''

############# Codimension 2 bifurcation diagrams

# tau and eta

fig2, ax = plt.subplots(figsize=(3.5,1.8), dpi=dpi, ncols=1)

# plot eta-tau continuation of the limit cycle

ax = a.plot_continuation('PAR(1)', 'PAR(4)', cont='tau_r/eta/hb2', ax=ax, ignore=['LP', 'BP', 'UZ'],
                         line_style_unstable='dashed', default_size=markersize1)

ax = a.plot_continuation('PAR(1)', 'PAR(3)', cont='tau_r/eta/gh1', ax=ax, ignore=['LP', 'BP', 'UZ'],
                         line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E', line_style_unstable='solid',
                         default_size=markersize1)

ax = a.plot_continuation('PAR(1)', 'PAR(4)', cont='tau_r/eta/lc_pd1', ax=ax, ignore=['LP', 'BP', 'UZ', 'R1', 'R2'],
                         line_color_stable='#148F77', line_color_unstable='#148F77', line_style_unstable='dotted',
                         default_size=markersize1)
ax.set_ylim((-15,15))

# cosmetics
ax.set_xlabel(r'$\bar\eta$')
ax.set_ylabel(r'$\tau$')
# ax.set_xticks(xticklabels[::3])
# ax.set_yticks(yticklabels[::3])
# ax.set_xlim([-6.5, -2.5])
# ax.set_ylim([0., 0.2])
ax.set_title('2D Limit Cycle Continuation')
plt.tight_layout()
#plt.savefig('fig2.svg')

plt.show()



#     # plot eta continuation for different alphas
#     ############################################
#
#     uv = 1
#     ax = ax_data[0]
#     for s in solutions_eta:
#
#         # plot the principal continuation
#         results, stability = get_cont_results(s, data_idx=[0, uv + 1])
#         col = get_line_collection(results[:, 1], results[:, 0], stability, c=['#76448A', '#5D6D7E'],
#                                   linethickness=linewidth)
#         ax.add_collection(col)
#
#         # plot the bifurcation y
#         plt.sca(ax)
#         hbs, lps = s('HB'), s('LP')
#         for l in lps:
#             plt.scatter(l['PAR(1)'], l['U(' + str(uv) + ')'], s=markersize2, marker='v', c='#5D6D7E')
#         for h in hbs:
#             plt.scatter(h['PAR(1)'], h['U(' + str(uv) + ')'], s=markersize2, marker='o', c='#148F77')
#
#     ax.set_xlim(-12.0, -2.0)
#     ax.set_ylim(0., 1.5)
#     ax.set_xlabel(r'$\eta$')
#     ax.set_ylabel('Firing rate (r)')
#
# if bifurcation_analysis and fixed_point_analysis:
#
#     uvs = [1]
#     uv_names = ['Firing rate (r)', 'v', 'e', 'a']
#     xlims = [(-6.0, -5.0), (), (), ()]
#     ylims = [(0.0, 2.5), (), (), ()]
#     for uv, name, xl, yl in zip(uvs, uv_names, xlims, ylims):
#
#         # plot the principal continuation
#         results, stability = get_cont_results(se, data_idx=[0, uv+1])
#         col = get_line_collection(results[:, 1], results[:, 0], stability, c=['#76448A', '#5D6D7E'],
#                                   linethickness=linewidth)
#         ax = ax_data[1]
#         ax.add_collection(col)
#
#         # plot the bifurcation y
#         plt.sca(ax)
#         hbs, lps, shs = se('HB'), se('LP'), se_hb2('LP')
#         for l in lps:
#             plt.scatter(l['PAR(1)'], l['U(' + str(uv) + ')'], s=markersize2, marker='v', c='#5D6D7E')
#         for h in hbs:
#             plt.scatter(h['PAR(1)'], h['U(' + str(uv) + ')'], s=markersize2, marker='o', c='#148F77')
#         for s in shs:
#             y = s['U(' + str(uv) + ')']
#             plt.scatter(s['PAR(1)'], np.max(y), s=markersize2, marker='p', c='#5D6D7E')
#             plt.scatter(s['PAR(1)'], np.min(y), s=markersize2, marker='p', c='#5D6D7E')
#
#         # plot the limit cycle of the first hopf bifurcation
#         results_hb1, stability_hb1, solutions_hb1, points_hb1 = get_cont_results(se_hb1, data_idx=[0, uv+1],
#                                                                                  extract_solution=[uv, 11])
#         p_min_hb1, p_max_hb1, period_hb1 = [], [], []
#         for s in solutions_hb1:
#             p_min_hb1.append(np.min(s[0]))
#             p_max_hb1.append(np.max(s[0]))
#             period_hb1.append(s[1])
#         col_min_hb1 = get_line_collection(np.array(p_min_hb1), results_hb1[points_hb1, 0], stability_hb1[points_hb1],
#                                           c=['#148F77', 'k'], linethickness=linewidth)
#         col_max_hb1 = get_line_collection(np.array(p_max_hb1), results_hb1[points_hb1, 0], stability_hb1[points_hb1],
#                                           c=['#148F77', 'k'], linethickness=linewidth)
#         ax.add_collection(col_min_hb1)
#         ax.add_collection(col_max_hb1)
#
#         # repeat for the second hopf bifurcation
#         results_hb2, stability_hb2, solutions_hb2, points_hb2 = get_cont_results(se_hb2, data_idx=[0, uv+1],
#                                                                                  extract_solution=[uv, 11])
#         p_min_hb2, p_max_hb2, period_hb2 = [], [], []
#         for s in solutions_hb2:
#             p_min_hb2.append(np.min(s[0]))
#             p_max_hb2.append(np.max(s[0]))
#             period_hb2.append(s[1])
#         p_min_hb2 = np.asarray(p_min_hb2)
#         p_max_hb2 = np.asarray(p_max_hb2)
#         col_min_hb2 = get_line_collection(np.array(p_min_hb2), results_hb2[points_hb2, 0], stability_hb2[points_hb2],
#                                           c=['#148F77', 'k'], linethickness=linewidth)
#         col_max_hb2 = get_line_collection(np.array(p_max_hb2), results_hb2[points_hb2, 0], stability_hb2[points_hb2],
#                                           c=['#148F77', 'k'], linethickness=linewidth)
#         ax.add_collection(col_min_hb2)
#         ax.add_collection(col_max_hb2)
#         ax.set_xlim(xl[0], xl[1])
#         ax.set_ylim(yl[0], yl[1])
#         ax.set_xlabel(r'$\eta$')
#         #ax.set_ylabel(name)
#
# plt.tight_layout()
# plt.savefig('fig2a.svg')
#
# # visualization of codimension 2 stuff
# ######################################
#
# if limit_cycle_analysis:
#
#     fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(4, 5), dpi=dpi)
#
#     results_lcb1, stability_lcb1 = get_cont_results(se_lcb1, data_idx=[0, -1])
#     results_lcb2, stability_lcb2 = get_cont_results(se_lcb2, data_idx=[0, -1])
#     col_lcb1 = get_line_collection(results_lcb1[:, 1], results_lcb1[:, 0], stability_lcb1, c='k',
#                                    linestyles=['-', '--'], linethickness=linewidth)
#     col_lcb2 = get_line_collection(results_lcb2[:, 1], results_lcb2[:, 0], stability_lcb2, c='k',
#                                    linestyles=['-', '--'], linethickness=linewidth)
#
#     results_lcs1, stability_lcs1, solution_lcs1, points_lcs1 = get_cont_results(se_lcs_upper, data_idx=[0, -1],
#                                                                                 extract_solution=[1, 2, 3, 11])
#     #results_lcs2, stability_lcs2, solution_lcs2, points_lcs2 = get_cont_results(se_lcs_lower, data_idx=[0, -1],
#     #                                                                            extract_solution=[1, 2, 3, 11])
#     col_lcs1 = get_line_collection(results_lcs1[:, 1], results_lcs1[:, 0], stability_lcs1, c='#148F77',
#                                    linestyles=['-', '-'], linethickness=linewidth)
#     #col_lcs2 = get_line_collection(results_lcs2[:, 1], results_lcs2[:, 0], stability_lcs2, c='#148F77',
#     #                               linestyles=['-', '-'], linethickness=linewidth)
#
#     results_fp1, stability_fp1 = get_cont_results(se_fp1, data_idx=[0, -1])
#     results_fp2, stability_fp2 = get_cont_results(se_fp2, data_idx=[0, -1])
#     col_fp1 = get_line_collection(results_fp1[:, 1], results_fp1[:, 0], stability_fp1, c='#5D6D7E',
#                                   linestyles=['-', '--'], linethickness=linewidth)
#     col_fp2 = get_line_collection(results_fp2[:, 1], results_fp2[:, 0], stability_fp2, c='#5D6D7E',
#                                   linestyles=['-', '--'], linethickness=linewidth)
#
#     # plot the lines
#     ax2[0, 0].add_collection(col_lcb1)
#     ax2[0, 0].add_collection(col_lcb2)
#     ax2[0, 0].add_collection(col_lcs1)
#     #ax2[0, 0].add_collection(col_lcs2)
#     ax2[0, 0].add_collection(col_fp1)
#     ax2[0, 0].add_collection(col_fp2)
#
#     # plot the bifurcation y
#     plt.sca(ax2[0, 0])
#     bts, ghs, cps = se_lcb1('BT') + se_lcb2('BT'), se_lcb1('GH') + se_lcb2('GH'), se_fp1('CP') + se_fp2('CP')
#     for b in bts:
#         plt.scatter(b['PAR(1)'], b['PAR(5)'], s=markersize1, marker='s', c='k')
#     for g in ghs:
#         plt.scatter(g['PAR(1)'], g['PAR(5)'], s=markersize1, marker='o', c='#148F77')
#     for c in cps:
#         plt.scatter(c['PAR(1)'], c['PAR(5)'], s=markersize1, marker='d', c='#5D6D7E')
#     ax2[0, 0].autoscale()
#     #ax2[0].set_xlim(-7, -2.5)
#     #ax2[0].set_ylim(0., 0.17)
#     #ax2[0, 0].set_xticks(ax2[0, 0].get_xticks()[::3])
#     #ax2[0, 0].set_yticks(ax2[0, 0].get_yticks()[::2])
#     ax2[0, 0].set_xlabel(r'$\eta$')
#     ax2[0, 0].set_ylabel(r'$\tau_d$')
#     #ax2[0, 0].set_title('Codimension 2 bifurcations')
#
#     # visualization of limit cycle period
#     plt.sca(ax2[0, 1])
#     im = plt.imshow(period_solutions, aspect='auto', interpolation='nearest', cmap='magma')
#     divider = make_axes_locatable(ax2[0, 1])
#     cax = divider.append_axes("right", size="8%", pad=0.1)
#     plt.colorbar(im, cax=cax, label='units of tau')
#     ax2[0, 1].grid(False)
#     ax2[0, 1].set_xlabel(r'$\eta$')
#     ax2[0, 1].set_ylabel(r'$\tau_d$')
#     ax2[0, 1].set_xticks(np.arange(period_solutions.shape[1])[::20])
#     ax2[0, 1].set_yticks(np.arange(period_solutions.shape[0])[::20])
#     ax2[0, 1].set_xticklabels(np.round(etas[::20], decimals=1))
#     ax2[0, 1].set_yticklabels(np.round(tau2s[::20], decimals=1))
#     ax2[0, 1].set_title('Limit cycle period')
#
#     # same shit for eta-tau1
#     ########################
#
#     results_lcb3, stability_lcb3 = get_cont_results(se_lcb3, data_idx=[0, -1])
#     results_lcb4, stability_lcb4 = get_cont_results(se_lcb4, data_idx=[0, -1])
#     col_lcb3 = get_line_collection(results_lcb3[:, 1], results_lcb3[:, 0], stability_lcb3, c='k',
#                                    linestyles=['-', '--'], linethickness=linewidth)
#     col_lcb4 = get_line_collection(results_lcb4[:, 1], results_lcb4[:, 0], stability_lcb4, c='k',
#                                    linestyles=['-', '--'], linethickness=linewidth)
#
#     results_lcs3, stability_lcs3, solution_lcs3, points_lcs3 = get_cont_results(se_lcs_upper2, data_idx=[0, -1],
#                                                                                 extract_solution=[1, 2, 3, 11])
#     results_lcs4, stability_lcs4, solution_lcs4, points_lcs4 = get_cont_results(se_lcs_lower2, data_idx=[0, -1],
#                                                                                 extract_solution=[1, 2, 3, 11])
#     col_lcs3 = get_line_collection(results_lcs3[:, 1], results_lcs3[:, 0], stability_lcs3, c='#148F77',
#                                    linestyles=['-', '-'], linethickness=linewidth)
#     col_lcs4 = get_line_collection(results_lcs4[:, 1], results_lcs4[:, 0], stability_lcs4, c='#148F77',
#                                    linestyles=['-', '-'], linethickness=linewidth)
#
#     # plot the lines
#     ax2[1, 0].add_collection(col_lcb3)
#     ax2[1, 0].add_collection(col_lcb4)
#     ax2[1, 0].add_collection(col_lcs3)
#     ax2[0, 1].add_collection(col_lcs4)
#
#     # plot the bifurcation y
#     plt.sca(ax2[1, 0])
#     bts, ghs = se_lcb3('BT') + se_lcb4('BT'), se_lcb3('GH') + se_lcb4('GH')
#     #bts, ghs, cps = se_lcb3('BT') + se_lcb4('BT'), se_lcb3('GH') + se_lcb4('GH'), se_fp3('CP') + se_fp4('CP')
#     for b in bts:
#         plt.scatter(b['PAR(1)'], b['PAR(4)'], s=markersize1, marker='s', c='k')
#     for g in ghs:
#         plt.scatter(g['PAR(1)'], g['PAR(4)'], s=markersize1, marker='o', c='#148F77')
#     ax2[1, 0].autoscale()
#     ax2[1, 0].set_xlim(-8.5, -2.5)
#     ax2[1, 0].set_ylim(0.0, 3.8)
#     ax2[1, 0].set_yticks(ax2[1, 0].get_yticks()[::2])
#     ax2[1, 0].set_xlabel(r'$\eta$')
#     ax2[1, 0].set_ylabel(r'$\tau_r$')
#
#     # same shit for tau1-tau2
#     #########################
#
#     results_lcb1, stability_lcb1 = get_cont_results(se_lcb5, data_idx=[0, -1])
#     results_lcb2, stability_lcb2 = get_cont_results(se_lcb6, data_idx=[0, -1])
#     col_lcb1 = get_line_collection(results_lcb1[:, 1], results_lcb1[:, 0], stability_lcb1, c='k',
#                                    linestyles=['-', '--'], linethickness=linewidth)
#     col_lcb2 = get_line_collection(results_lcb2[:, 1], results_lcb2[:, 0], stability_lcb2, c='k',
#                                    linestyles=['-', '--'], linethickness=linewidth)
#
#     results_lcs1, stability_lcs1, solution_lcs1, points_lcs1 = get_cont_results(se_lcs_upper3, data_idx=[0, -1],
#                                                                                 extract_solution=[1, 2, 3, 11])
#     #results_lcs2, stability_lcs2, solution_lcs2, points_lcs2 = get_cont_results(se_lcs_lower3, data_idx=[0, -1],
#     #                                                                            extract_solution=[1, 2, 3, 11])
#     col_lcs1 = get_line_collection(results_lcs1[:, 1], results_lcs1[:, 0], stability_lcs1, c='#148F77',
#                                    linestyles=['-', '-'], linethickness=linewidth)
#     #col_lcs2 = get_line_collection(results_lcs2[:, 1], results_lcs2[:, 0], stability_lcs2, c='#148F77',
#     #                               linestyles=['-', '-'], linethickness=linewidth)
#
#     # plot the lines
#     ax2[1, 1].add_collection(col_lcb1)
#     ax2[1, 1].add_collection(col_lcb2)
#     ax2[1, 1].add_collection(col_lcs1)
#     #ax2[1, 0].add_collection(col_lcs2)
#
#     # plot the bifurcation y
#     plt.sca(ax2[1, 1])
#     bts, ghs = se_lcb5('BT') + se_lcb6('BT'), se_lcb5('GH') + se_lcb6('GH')
#     for b in bts:
#         plt.scatter(b['PAR(5)'], b['PAR(4)'], s=markersize1, marker='s', c='k')
#     for g in ghs:
#         plt.scatter(g['PAR(5)'], g['PAR(4)'], s=markersize1, marker='o', c='#148F77')
#     ax2[1, 1].autoscale()
#     ax2[1, 1].set_xlim(0., 25.0)
#     ax2[1, 1].set_ylim(0., 2.2)
#     #ax2[1, 1].set_xticks(ax2[1, 1].get_xticks()[::3])
#     #ax2[1, 1].set_yticks(ax2[1, 1].get_yticks()[::2])
#     ax2[1, 1].set_xlabel(r'$\tau_d$')
#     ax2[1, 1].set_ylabel(r'$\tau_r$')
#     #ax2[1, 0].set_title('Codimension 2 bifurcations')
#
#     plt.tight_layout()
#     plt.savefig('fig4.svg')
#
# plt.show()



'''
# load pyauto instance from file
fname = 'results/biexp_mult_strange_attractor.pkl'
a = PyAuto.from_file(fname)

chaos_data = a.additional_attributes['chaos_analysis']

fig = plt.figure()


etas = chaos_data['eta']
eta_rounded = list(map(lambda x : round(x,2), chaos_data['eta'])) 
lps = chaos_data['lyapunov_exponents']
fract_dim = chaos_data['fractal_dim']

n_per_row = 4
n_rows = int(np.ceil(len(etas)/n_per_row))
gs = fig.add_gridspec(n_rows, n_per_row)

row = 0
col = 0
for i, elem in enumerate(eta_rounded):
    f_ax1 = fig.add_subplot(gs[row, col], projection='3d')
    ax = a.plot_trajectory(['U(1)', 'U(2)', 'U(3)'], ax=f_ax1, cont=f'eta_{i+1}_t', force_axis_lim_update=True, cmap=plt.get_cmap('magma'),cutoff=200)
    max_LP = round(max(chaos_data['lyapunov_exponents'][i]), 3)
    fract_dim = round(chaos_data['fractal_dim'][i], 3)
    ax.set_title(fr'$\eta$ ={elem}'+f'\nMax LP: {max_LP}'+f'\nDim: {fract_dim}')
    col+=1
    if not (i+1)%n_per_row:
        row+=1
        col=0

imagepath='../../plots/'+f'biexp_mult_{eta_rounded}'+'.png'
plt.savefig(imagepath)
plt.show()
'''
