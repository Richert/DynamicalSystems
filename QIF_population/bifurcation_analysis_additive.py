import os
import auto as a
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import numpy as np
os.chdir("auto_files")
plt.ioff()

plt.style.reload_library()
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['legend.fontsize'] = 20

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

    # combine points and param vals
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


#####################################
# bifurcations for eta continuation #
#####################################

# numerical continuations
#########################

# initial continuation
"""This serves the purpose of setting the adaptation strength to a desired initial value > 0 using a specified tau.
"""
alpha_0 = [0.01, 0.02, 0.04, 0.08, 0.16]
s0 = a.run(e='qif2', c='qif', ICP=3, UZR={3: alpha_0}, STOP=['UZ'+str(len(alpha_0))], DSMAX=0.005)

# primary parameter continuation in eta
"""Now, we look for a bifurcation when increasing the excitability of the system via eta.
"""
se_col = []
se_col.append(a.merge(a.run(s0('EP1'), ICP=1, DSMAX=0.005, RL0=-12.0, NMX=2000) +
                      a.run(s0('EP1'), ICP=1, DSMAX=0.005, RL0=-12.0, DS='-', NMX=2000)))
for s in s0('UZ'):
    se_col.append(a.merge(a.run(s, ICP=1, DSMAX=0.005, RL0=-12.0, NMX=8000) +
                          a.run(s, ICP=1, DSMAX=0.005, RL0=-12.0, DS='-', NMX=8000)))
se = se_col[3]

if fixed_point_analysis:

    # limit cycle continuation of hopf bifurcations in eta
    """Now, we continue the limit cycle of the first and second hopf bifurcation in eta.
    """
    se_hb1 = a.run(se('HB1'), c='qif2b', ICP=1, DSMAX=0.1, NMX=2000)
    se_hb2 = a.run(se('HB2'), c='qif2b', ICP=1, DSMAX=0.1, NMX=2000, UZR={1: -4.9}, STOP={})

# visualization of eta continuation
###################################

if bifurcation_analysis:

    # plot eta continuation for different alphas
    ############################################

    uv = 1
    fig, ax = plt.subplots(figsize=(15, 8))
    for s in se_col:

        # plot the principal continuation
        results, stability = get_cont_results(s, data_idx=[0, uv + 1])
        col = get_line_collection(results[:, 1], results[:, 0], stability, c=['#76448A', '#5D6D7E'])
        ax.add_collection(col)
        ax.autoscale()

        # plot the bifurcation points
        hbs, lps = s('HB'), s('LP')
        for l in lps:
            plt.scatter(l['PAR(1)'], l['U(' + str(uv) + ')'], s=200, marker='v', c='#5D6D7E')
        for h in hbs:
            plt.scatter(h['PAR(1)'], h['U(' + str(uv) + ')'], s=200, marker='o', c='#148F77')

if bifurcation_analysis and fixed_point_analysis:

    uvs = [1]
    uv_names = ['r', 'v', 'e', 'a']
    xlims = [(-6.5, -4), (), (), ()]
    ylims = [(0.0, 2.5), (), (), ()]
    for uv, name, xl, yl in zip(uvs, uv_names, xlims, ylims):

        # plot the principal continuation
        results, stability = get_cont_results(se, data_idx=[0, uv+1])
        col = get_line_collection(results[:, 1], results[:, 0], stability, c=['#76448A', '#5D6D7E'], linethickness=2.0)
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.add_collection(col)
        ax.autoscale()

        # plot the bifurcation points
        hbs, lps = se('HB'), se('LP')
        for l in lps:
            plt.scatter(l['PAR(1)'], l['U(' + str(uv) + ')'], s=300, marker='v', c='#5D6D7E')
        for h in hbs:
            plt.scatter(h['PAR(1)'], h['U(' + str(uv) + ')'], s=300, marker='o', c='#148F77')

        # plot the limit cycle of the first hopf bifurcation
        results_hb1, stability_hb1, solutions_hb1, points_hb1 = get_cont_results(se_hb1, data_idx=[0, uv+1],
                                                                                 extract_solution=[uv, 11])
        p_min_hb1, p_max_hb1, period_hb1 = [], [], []
        for s in solutions_hb1:
            p_min_hb1.append(np.min(s[0]))
            p_max_hb1.append(np.max(s[0]))
            period_hb1.append(s[1])
        col_min_hb1 = get_line_collection(np.array(p_min_hb1), results_hb1[points_hb1, 0], stability_hb1[points_hb1],
                                          c=['#148F77', 'k'], linethickness=2.0)
        col_max_hb1 = get_line_collection(np.array(p_max_hb1), results_hb1[points_hb1, 0], stability_hb1[points_hb1],
                                          c=['#148F77', 'k'], linethickness=2.0)
        ax.add_collection(col_min_hb1)
        ax.add_collection(col_max_hb1)

        # repeat for the second hopf bifurcation
        results_hb2, stability_hb2, solutions_hb2, points_hb2 = get_cont_results(se_hb2, data_idx=[0, uv+1],
                                                                                 extract_solution=[uv, 11])
        p_min_hb2, p_max_hb2, period_hb2 = [], [], []
        for s in solutions_hb2:
            p_min_hb2.append(np.min(s[0]))
            p_max_hb2.append(np.max(s[0]))
            period_hb2.append(s[1])
        p_min_hb2 = np.asarray(p_min_hb2)
        p_max_hb2 = np.asarray(p_max_hb2)
        col_min_hb2 = get_line_collection(np.array(p_min_hb2), results_hb2[points_hb2, 0], stability_hb2[points_hb2],
                                          c=['#148F77', 'k'], linethickness=2.0)
        col_max_hb2 = get_line_collection(np.array(p_max_hb2), results_hb2[points_hb2, 0], stability_hb2[points_hb2],
                                          c=['#148F77', 'k'], linethickness=2.0)
        ax.add_collection(col_min_hb2)
        ax.add_collection(col_max_hb2)
        ax.autoscale()
        #ax.set_xlim(xl[0], xl[1])
        #ax.set_ylim(yl[0], yl[1])
        ax.set_xlabel('eta')
        ax.set_ylabel(name)

        # shade the area between the min/max of the limit cycle
        #ax.fill_between(x=results_hb2[:, 0], y1=p_min_hb2, y2=p_max_hb2, where=stability_hb2 < 0., alpha=0.3)
        #ax.fill_between(x=results_hb2[:, 0], y1=p_min_hb2, y2=p_max_hb2, where=stability_hb2 < 0., alpha=0.3)

plt.show()
