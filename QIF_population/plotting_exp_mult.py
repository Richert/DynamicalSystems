import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection

plt.ioff()

# plotting parameters
linewidth = 0.5
fontsize1 = 6
fontsize2 = 8
markersize1 = 30
markersize2 = 30
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

# visualization of eta continuation
###################################

fig, axes = plt.subplots(ncols=2, figsize=(7, 1.8), dpi=dpi)

# plot eta continuation for different alphas
uv = 1
ax = axes[0]
for s in se_col:

    # plot the principal continuation
    results, stability = get_cont_results(s, data_idx=[0, uv + 1])
    col = get_line_collection(results[:, 1], results[:, 0], stability, c=['#76448A', '#5D6D7E'],
                              linethickness=linewidth)
    ax.add_collection(col)

    # plot the bifurcation y
    plt.sca(ax)
    hbs, lps = s('HB'), s('LP')
    for l in lps:
        plt.scatter(l['PAR(1)'], l['U(' + str(uv) + ')'], s=markersize2, marker='v', c='#5D6D7E')
    for h in hbs:
        plt.scatter(h['PAR(1)'], h['U(' + str(uv) + ')'], s=markersize2, marker='o', c='#148F77')

ax.set_xlim(-12.0, 2.0)
ax.set_xlabel(r'$\eta$')
ax.set_ylabel('Firing rate (r)')

uvs = [1]
uv_names = ['Firing rate (r)', 'v', 'e']
xlims = [(-10, 2), (), (), ()]
ylims = [(0.0, 2.5), (), (), ()]
for uv, name, xl, yl in zip(uvs, uv_names, xlims, ylims):

    # plot the principal continuation
    results, stability = get_cont_results(se, data_idx=[0, uv+1])
    col = get_line_collection(results[:, 1], results[:, 0], stability, c=['#76448A', '#5D6D7E'],
                              linethickness=linewidth)
    ax = axes[1]
    ax.add_collection(col)

    # plot the bifurcation y
    plt.sca(ax)
    hbs, lps = se('HB'), se('LP')
    for l in lps:
        plt.scatter(l['PAR(1)'], l['U(' + str(uv) + ')'], s=markersize2, marker='v', c='#5D6D7E')
    for h in hbs:
        plt.scatter(h['PAR(1)'], h['U(' + str(uv) + ')'], s=markersize2, marker='o', c='#148F77')

    # plot the limit cycle of the first hopf bifurcation
    results_hb1, stability_hb1, solutions_hb1, points_hb1 = get_cont_results(se_hb1, data_idx=[0, uv+1],
                                                                             extract_solution=[uv, 11])
    p_min_hb1, p_max_hb1, period_hb1 = [], [], []
    for s in solutions_hb1:
        p_min_hb1.append(np.min(s[0]))
        p_max_hb1.append(np.max(s[0]))
        period_hb1.append(s[1])
    col_min_hb1 = get_line_collection(np.array(p_min_hb1), results_hb1[points_hb1, 0], stability_hb1[points_hb1],
                                      c=['#148F77', 'k'], linethickness=linewidth)
    col_max_hb1 = get_line_collection(np.array(p_max_hb1), results_hb1[points_hb1, 0], stability_hb1[points_hb1],
                                      c=['#148F77', 'k'], linethickness=linewidth)
    ax.add_collection(col_min_hb1)
    ax.add_collection(col_max_hb1)

    # plot the second limit cycle, including all period doubling bifurcations
    for se_pd in se_pds:
        results_hb2, stability_hb2, solutions_hb2, points_hb2 = get_cont_results(se_pd, data_idx=[0, uv+1],
                                                                                 extract_solution=[uv, 11])
        p_min_hb2, p_max_hb2, period_hb2 = [], [], []
        for s in solutions_hb2:
            p_min_hb2.append(np.min(s[0]))
            p_max_hb2.append(np.max(s[0]))
            period_hb2.append(s[1])
        p_min_hb2 = np.asarray(p_min_hb2)
        p_max_hb2 = np.asarray(p_max_hb2)
        col_min_hb2 = get_line_collection(np.array(p_min_hb2), results_hb2[points_hb2, 0], stability_hb2[points_hb2],
                                          c=['#148F77', 'k'], linethickness=linewidth)
        col_max_hb2 = get_line_collection(np.array(p_max_hb2), results_hb2[points_hb2, 0], stability_hb2[points_hb2],
                                          c=['#148F77', 'k'], linethickness=linewidth)
        ax.add_collection(col_min_hb2)
        ax.add_collection(col_max_hb2)

        for pd in se_pd('PD'):
            y = pd['U(' + str(uv) + ')']
            plt.scatter(pd['PAR(1)'], np.max(y), s=markersize2, marker='h', c='#5D6D7E')
            plt.scatter(pd['PAR(1)'], np.min(y), s=markersize2, marker='h', c='#5D6D7E')

    ax.set_xlim(xl[0], xl[1])
    ax.set_ylim(yl[0], yl[1])
    ax.set_xlabel(r'$\eta$')

plt.tight_layout()
#plt.savefig('fig2a.svg')

if pd_analysis:

    # visualization of period doublings
    ###################################

    # 3d plot of pp for eta, alpha and e
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    for spp in solutions_pp:
        ax2.plot(xs=spp[1], ys=spp[0], zs=spp[2], linewidth=2.0, alpha=0.7)
    ax2.tick_params(axis='both', which='major', pad=20)
    ax2.set_xlabel('v', labelpad=30)
    ax2.set_ylabel('r', labelpad=30)
    ax2.set_zlabel('e', labelpad=30)

    # visualization of periods
    fig3, ax3 = plt.subplots(figsize=(5, 2), dpi=dpi)
    plt.sca(ax3)
    plt.scatter(etas_pd, periods_pd)
    ax3.set_xlabel(r'$\eta$')
    ax3.set_ylabel('period')

plt.show()
