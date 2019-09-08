import os
import auto as a
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
import numpy as np
os.chdir("auto_files")
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
mpl.rcParams['axes.titlesize'] = fontsize2
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = fontsize2
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = fontsize1
mpl.rcParams['ytick.labelsize'] = fontsize1
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['legend.fontsize'] = fontsize1
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
mpl.rc('text', usetex=True)

#########################################
# configs, descriptions and definitions #
#########################################

# problem description
#####################

"""
Performs continuation of extended Montbrio population given the initial condition:

 U1: r_0 = 0.114741, 
 U2: v_0 = -2.774150,
 U3: e_0 = 0.0,
 U4: a_0 = 0.0

with parameters:

 PAR1: eta = -10.0
 PAR2: J = 15.0
 PAR3: alpha = 0.0
 PAR4: tau = 1.0
 PAR5: D = 2.0

HOWTO:
 - Below, indicate the continuation parameters by using the keyword ICP and the indices of the respective parameters.
 - Use the keywords RL0 and RL1 to set the boundaries between which to perform the continuation.
 - Use UZR={PARIDX:PARVAL} to indicate a user-specified stopping point for the continuation (via the same syntax,
   the keyword PAR can be used to change parameter values).
 - To continue from a previous solution, pass a previous solution as first argument to run() and specify the solution 
   point to start from via the keyword IRS (e.g. IRS='LP1' where 1 is the index)
 - Use Dc to specify the continuation direction (e.g. '-')
 - Specify ISW to control branch switching
 - Specify IPS to set the problem type (1=Stationary solution of ODEs, -2=Time integration via Implicit Euler, 
   2=Periodic solution of ODEs, 9=Detection of homoclinic bifurcations) 
"""


# configuration
###############

bifurcation_analysis = True
fixed_point_analysis = True
pd_analysis = True
limit_cycle_analysis = True

# definitions
#############


def get_cont_results(s, data_idx=[1], extract_solution=[]):
    """

    :param s:
    :param state_var_idx
    :param param_idx
    :param extract_solution
    :return:
    """

    solution = s()
    data = s[0].coordarray
    results = np.zeros((data.shape[1], len(data_idx)))
    point_solutions = []
    points = []
    stability = np.zeros((results.shape[0],))
    start = 0
    for i, s in enumerate(solution):
        end = s['PT']
        for j, idx in enumerate(data_idx):
            results[start:end, j] = data[idx, start:end]
        stability[start:end] = s.b['solution'].b['PT'] < 0
        if extract_solution:
            solutions = []
            for idx in extract_solution:
                if idx == 11:
                    solutions.append(s.b['solution']['PAR(11)'])
                else:
                    solutions.append(s.b['solution']['U(' + str(idx) + ')'])
            points.append(end - 1)
            point_solutions.append(solutions)
        start = end
    stability = stability[:start]
    if point_solutions:
        return results, stability, point_solutions, points
    return results, stability


def get_line_collection(points, param_vals, stability, c='r', linestyles=['-', '--'], linethickness=1.0):
    """

    :param points:
    :param param_vals:
    :param stability:
    :param c
    :param linestyles
    :param linethickness
    :return:
    """

    if type(c) is str:
        c = [c, c]

    # combine y and param vals
    points = np.reshape(points, (points.shape[0], 1))
    param_vals = np.reshape(param_vals, (len(param_vals), 1))
    points = np.append(param_vals, points, axis=1)

    # get indices for line segments
    idx = np.asarray(stability == 1)
    idx_d = np.concatenate([np.zeros((1,)), np.diff(idx)])
    idx_s = np.sort(np.argwhere(idx_d == 1))
    idx_s = np.append(idx_s, len(idx))

    # create line segments
    lines = []
    styles = []
    colors = []
    style_idx = 0 if idx[0] else 1
    point_idx = 1
    for i in idx_s:
        lines.append(points[point_idx-1:i, :])
        styles.append(linestyles[style_idx])
        colors.append(c[style_idx])
        style_idx = abs(style_idx - 1)
        point_idx = i
    return LineCollection(lines, linestyles=styles, colors=colors, linewidths=linethickness)


def extract_from_solution(s, params, vars, auto_params=None, time_vars=None):
    """

    :param s:
    :param params:
    :param vars:
    :param time_vars:
    :return:
    """

    if hasattr(s, 'b'):
        sol = s.b['solution']
    else:
        sol = s

    param_col = [sol[p] for p in params]
    var_vol = [s[v] for v in vars]
    ts_col = [sol[v] for v in time_vars] if time_vars else []
    auto_par_col = [s.b[p] for p in auto_params] if auto_params else []

    return param_col, var_vol, auto_par_col, ts_col


def continue_period_doubling_bf(s, max_iter=100, iter=0):
    """
    :param s:
    :return:
    """
    solutions = []
    for pd in s('PD'):
        s_tmp = a.run(pd, c='qif2b', ICP=[1, 11], NMX=2000, NTST=600, DSMAX=0.05, ILP=0, NDIM=3)
        solutions.append(s_tmp)
        if iter >= max_iter:
            break
        else:
            solutions += continue_period_doubling_bf(s_tmp, iter=iter+1)
    return solutions


def get_lyapunov_exponent(bf_diag, branches, points):
    """

    :param bf_diag:
    :param branches:
    :param points:
    :return:
    """

    diag = bf_diag.data[0].diagnostics.data
    N = len(diag)
    idx = 0
    LEs = []

    # go through target auto_solutions
    for b, p in zip(branches, points):

        # go through auto_solutions of diagnostic data
        for point_idx in range(idx, N):

            # extract relevant diagnostic text output
            diag_tmp = diag[point_idx]['Text'].split('\n\n')

            if len(diag_tmp) > 2:

                diag_tmp2 = diag_tmp[1].split('\n')

                # check whether branch and point identifiers match the targets
                target_str = str(b) + '    ' + str(p)
                if target_str in diag_tmp2[0]:

                    # extract period of solution
                    period = float(diag_tmp[2].split(' ')[-1])

                    # extract floquet multiplier (real parts)
                    lyapunovs = []
                    for i, floquet_line in enumerate(diag_tmp2[2:]):

                        # extract multipliers from string
                        multiplier = 'Multiplier  ' + str(i+1) + '   '
                        if multiplier in floquet_line:
                            start = floquet_line.index(multiplier) + len(multiplier)
                            stop = floquet_line.index('  Abs. Val.')
                            floquet_split = floquet_line[start:stop].split(' ')
                            floquet_real, floquet_imag = floquet_split[:-3][-1], floquet_split[-1]
                            floquet = complex(float(floquet_real), float(floquet_imag))
                        else:
                            break

                        # calculate lyapunov exponent from real part of floquet multipliers
                        lyapunovs.append(np.log(floquet)/period)

                    # set new starting point for search in solution diagnostics
                    LEs.append(lyapunovs)
                    idx = point_idx+1
                    break

    return LEs

#####################################
# bifurcations for eta continuation #
#####################################

# numerical continuations
#########################

# initial continuation
"""This serves the purpose of setting the adaptation strength to a desired initial value > 0 using a specified tau.
"""
alpha_0 = [0.0125, 0.025, 0.05, 0.1, 0.2]
s0 = a.run(e='qif4', c='qif4', ICP=3, UZR={3: alpha_0}, STOP=['UZ'+str(len(alpha_0))], DSMAX=0.005)

# primary parameter continuation in eta
"""Now, we look for a bifurcation when increasing the excitability of the system via eta.
"""
se_col = []
se_col.append(a.merge(a.run(s0('EP1'), ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0) +
                      a.run(s0('EP1'), ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, DS='-')))
for s in s0('UZ'):
    se_col.append(a.merge(a.run(s, ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0) +
                          a.run(s, ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, DS='-')))
se = se_col[3]

# limit cycle continuation of hopf bifurcations in eta
"""Now, we continue the limit cycle of the first and second hopf bifurcation in eta.
"""
se_hb1 = a.run(se('HB1'), c='qif2b', ICP=[1, 11], DSMAX=0.05, NMX=3000, NDIM=3)
se_hb2 = a.run(se('HB2'), c='qif2b', ICP=[1, 11], DSMAX=0.05, NMX=3000, NDIM=3)
get_lyapunov_exponent(se_hb2, [27], [56])

if pd_analysis:

    # continue the stable region of the limit cycle in eta
    """Here, we continue the period doubling bifurcations on the limit cycle.
    """

    se_pds = continue_period_doubling_bf(se_hb2)

    # extract the time-series and periods from period doubling cascades
    """Here, we investigate the trajectories in the r-v-e space given different initial conditions.
    """
    solutions_pp = []
    periods_pd = []
    etas_pd = []
    for s in se_pds:
        for pd in s('PD'):
            params, _, _, results_t = extract_from_solution(pd, params=['PAR(1)', 'PAR(11)'], vars=[],
                                                            time_vars=['U(1)', 'U(2)', 'U(3)'])
            solutions_pp.append(results_t)
            etas_pd.append(params[0])
            periods_pd.append(params[1])

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
