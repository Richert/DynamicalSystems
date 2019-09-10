import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pyauto import PyAuto

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
mpl.rcParams['ax_data.titlesize'] = fontsize2
mpl.rcParams['ax_data.titleweight'] = 'bold'
mpl.rcParams['ax_data.labelsize'] = fontsize2
mpl.rcParams['ax_data.labelcolor'] = 'black'
mpl.rcParams['ax_data.labelweight'] = 'bold'
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

fname = 'auto_files/biexp_mult.hdf5'
a = PyAuto.from_file(fname)
#period_solutions = np.fromfile(f"{fname}_period")

############
# plotting #
############

# principle continuation in eta
###############################

fig, axes = plt.subplots(ncols=2, figsize=(7, 1.8), dpi=dpi)

# plot principle eta continuation for different alphas
ax = axes[0]
n_alphas = 5
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_0', ax=ax)
for i in range(2, n_alphas+2):
    ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_{i}', ax=ax)

# plot eta continuation for single alpha with limit cycle continuation
ax = axes[1]
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_4', ax=ax)
ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_hb2', ax=ax)
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
