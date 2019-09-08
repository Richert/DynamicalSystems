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
phase_portrait_analysis = True
limit_cycle_analysis = True
n_dim = 6
n_par = 7

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


#####################################
# bifurcations for eta continuation #
#####################################

# numerical continuations
#########################

# initial continuation
"""This serves the purpose of setting the adaptation strength to a desired initial value > 0 using a specified tau.
"""
alpha_0 = 0.05
beta_0 = [0.125, 0.25, 0.5, 1.0, 2.0]
s0 = a.run(e='qif4', c='qif', ICP=3, UZR={3: alpha_0}, STOP=['UZ1'], DSMAX=0.005, NDIM=n_dim, NPAR=n_par)
s1 = a.run(s0('UZ1'), ICP=4, UZR={4: beta_0}, STOP=['UZ'+str(len(beta_0))], DSMAX=0.005)

# primary parameter continuation in eta
"""Now, we look for a bifurcation when increasing the excitability of the system via eta.
"""
se_col = []
se_col.append(a.merge(a.run(s0('EP1'), ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0) +
                      a.run(s0('EP1'), ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, DS='-')))
for s in s1('UZ'):
    se_col.append(a.merge(a.run(s, ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0) +
                          a.run(s, ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, DS='-')))
se = se_col[2]

if fixed_point_analysis or phase_portrait_analysis:

    # limit cycle continuation of hopf bifurcations in eta
    """Now, we continue the limit cycle of the first and second hopf bifurcation in eta.
    """
    se_hb1 = a.run(se('HB1'), c='qif2b', ICP=1, DSMAX=0.1, NMX=2000, NDIM=n_dim, NPAR=n_par)
    se_hb2 = a.run(se('HB2'), c='qif2b', ICP=1, DSMAX=0.1, NMX=2000, UZR={1: -4.9}, STOP={}, NDIM=n_dim, NPAR=n_par)

if limit_cycle_analysis:

    # continue the limit cycle borders in alpha and eta
    """Here, we investigate for which values of eta and alpha the limit cycle can be found.
    """
    se_lcb1 = a.run(se('HB1'), c='qif2', ICP=[1, 3], DSMAX=0.01, NMX=2000, NDIM=n_dim, NPAR=n_par)
    se_lcb2 = a.run(se('HB1'), c='qif2', ICP=[1, 3], DSMAX=0.01, NMX=2000, DS='-', NDIM=n_dim, NPAR=n_par)

    # continue the stable region of the limit cycle
    """Here, we investigate the parameter space in eta and alpha for which the limit cycle is stable.
    """
    alphas = np.round(np.linspace(0.03, 0.16, 20)[::-1], decimals=4).tolist()
    se_lcs1 = a.run(a.run(se_hb2('LP1'), c='qif3', ICP=[1, 3], NMX=2000, RL1=0.0, DSMAX=0.05, UZR={3: alphas}, STOP={},
                          NDIM=n_dim, NPAR=n_par))
    se_lcs2 = a.run(a.run(se_hb2('LP1'), c='qif3', ICP=[1, 3], NMX=2000, DS='-', RL1=0.0, DSMAX=0.05, UZR={3: alphas},
                          STOP={}, NDIM=n_dim, NPAR=n_par))
    se_lcs3 = a.run(a.run(se_hb2('LP2'), c='qif3', ICP=[1, 3], NMX=4000, UZR={3: alphas}, STOP={}, RL1=0.0, DSMAX=0.05,
                          NDIM=n_dim, NPAR=n_par))
    se_lcs4 = a.run(a.run(se_hb2('LP2'), c='qif3', ICP=[1, 3], NMX=4000, DS='-', UZR={3: alphas}, STOP={}, RL1=0.0,
                          DSMAX=0.05, NDIM=n_dim, NPAR=n_par))
    se_lcs_upper = a.merge(se_lcs3 + se_lcs4)
    se_lcs_lower = a.merge(se_lcs1 + se_lcs2)

    # continue the fold bifurcation in eta and alpha
    """Here, we investigate the parameter space for which the fold bifurcation exists.
    """
    se_fp1 = a.run(se('LP2'), c='qif2', ICP=[3, 1], RL0=0.0, DSMAX=0.01, NMX=2000, NDIM=n_dim, NPAR=n_par)
    se_fp2 = a.run(se('LP2'), c='qif2', ICP=[3, 1], RL0=0.0, DSMAX=0.01, NMX=2000, DS='-', NDIM=n_dim, NPAR=n_par)

    # get limit cycle periods
    """Here, we create auto_solutions for grid y in the eta-alpha plane where the limit cycle is stable, to extract the 
    cycle periods later on.
    """
    etas = np.round(np.linspace(-5.8, -3.5, 20), decimals=4).tolist()
    period_solutions = np.zeros((len(alphas), len(etas)))
    for s_tmp in se_lcs_upper():
        if np.round(s_tmp['PAR(3)'], decimals=4) in alphas:
            s_tmp2 = a.run(s_tmp, c='qif', ICP=[1], UZR={1: etas}, STOP={}, EPSL=1e-6, EPSU=1e-6, EPSS=1e-4,
                           ILP=0, ISP=0, IPS=2, DS='-', NDIM=n_dim, NPAR=n_par)
            eta_old = etas[0]
            for s in s_tmp2('UZ'):
                params_tmp, _, _, _ = extract_from_solution(s, params=['PAR(1)', 'PAR(3)', 'PAR(11)'], vars=[])
                if params_tmp[0] >= eta_old:
                    idx_c = np.argwhere(np.round(params_tmp[0], decimals=4) == etas)
                    idx_r = np.argwhere(np.round(params_tmp[1], decimals=4) == alphas)
                    period_solutions[idx_r, idx_c] = params_tmp[2]
                    eta_old = params_tmp[0]

    # continue the limit cycle borders in alpha and tau
    """Here, we investigate for which values of eta and alpha the limit cycle can be found.
    """
    se_lcb3 = a.run(se('HB2'), c='qif2', ICP=[1, 4], DSMAX=0.01, NMX=4000, NDIM=n_dim, NPAR=n_par)
    se_lcb4 = a.run(se('HB2'), c='qif2', ICP=[1, 4], DSMAX=0.01, NMX=6000, DS='-', NDIM=n_dim, NPAR=n_par)

    # continue the stable region of the limit cycle
    """Here, we investigate the parameter space in eta and alpha for which the limit cycle is stable.
    """
    taus = np.round(np.linspace(5.0, 35.0, 20)[::-1], decimals=4).tolist()
    se_lcs1 = a.run(a.run(se_hb2('LP1'), c='qif3', ICP=[1, 4], NMX=4000, UZR={4: taus}, STOP={}, RL1=0.0, DSMAX=0.05,
                          NDIM=n_dim, NPAR=n_par))
    se_lcs2 = a.run(a.run(se_hb2('LP1'), c='qif3', ICP=[1, 4], NMX=4000, DS='-', RL1=0.0, DSMAX=0.05, UZR={4: taus},
                          STOP={}, NDIM=n_dim, NPAR=n_par))
    se_lcs3 = a.run(a.run(se_hb2('LP2'), c='qif3', ICP=[1, 4], NMX=4000, RL1=0.0, DSMAX=0.05, UZR={4: taus}, STOP={},
                          NDIM=n_dim, NPAR=n_par))
    se_lcs4 = a.run(a.run(se_hb2('LP2'), c='qif3', ICP=[1, 4], NMX=4000, DS='-', UZR={4: taus}, STOP={}, RL1=0.0,
                          DSMAX=0.05, NDIM=n_dim, NPAR=n_par))
    se_lcs_lower2 = a.merge(se_lcs3 + se_lcs4)
    se_lcs_upper2 = a.merge(se_lcs1 + se_lcs2)

    # continue the fold bifurcation in eta and alpha
    """Here, we investigate the parameter space for which the fold bifurcation exists.
    """
    se_fp3 = a.run(se('LP2'), c='qif2', ICP=[1, 4], RL1=0.0, DSMAX=0.01, NMX=2000, NDIM=n_dim, NPAR=n_par)
    se_fp4 = a.run(se('LP2'), c='qif2', ICP=[1, 4], RL1=0.0, DSMAX=0.01, NMX=2000, DS='-', NDIM=n_dim, NPAR=n_par)

    # get limit cycle periods
    """Here, we create auto_solutions for grid y in the eta-alpha plane where the limit cycle is stable, to extract the 
    cycle periods later on.
    """
    etas = np.round(np.linspace(-6.0, -3.0, 20), decimals=4).tolist()
    period_solutions2 = np.zeros((len(taus), len(etas)))
    for s_tmp in se_lcs_upper2():
        if np.round(s_tmp['PAR(4)'], decimals=4) in taus:
            s_tmp2 = a.run(s_tmp, c='qif', ICP=1, UZR={1: etas}, STOP={}, EPSL=1e-6, EPSU=1e-6, EPSS=1e-4,
                           ILP=0, ISP=0, IPS=2, DS='-', NDIM=n_dim, NPAR=n_par)
            eta_old = etas[0]
            for s in s_tmp2('UZ'):
                params_tmp, _, _, _ = extract_from_solution(s, params=['PAR(1)', 'PAR(4)', 'PAR(11)'], vars=[])
                if params_tmp[0] >= eta_old:
                    idx_c = np.argwhere(np.round(params_tmp[0], decimals=4) == etas)
                    idx_r = np.argwhere(np.round(params_tmp[1], decimals=4) == taus)
                    period_solutions2[idx_r, idx_c] = params_tmp[2]
                    eta_old = params_tmp[0]

if phase_portrait_analysis:

    # investigate the phase space around the bistable region in eta
    """Here, we investigate the trajectories in the r-v-e space given different initial conditions.
    """
    solutions_pp = []
    perturbations = [0.95, 1.05]
    dt = 0.001
    _, results, _, results_t = extract_from_solution(se_hb2('UZ1'), params=[],
                                                     vars=['MAX U(1)', 'MAX U(2)', 'MAX U(3)', 'MAX U(4)'],
                                                     time_vars=['U(1)', 'U(2)', 'U(3)', 'U(4)'])
    for p1 in perturbations:
        for p2 in perturbations:
            for p3 in perturbations:
                for p4 in perturbations:
                    for r0, v0, e0, a0 in zip(results_t[0][::70], results_t[1][::70], results_t[2][::70], results_t[3][::70]):
                        U = {1: r0*p1, 2: v0*p2, 3: e0*p3, 4: a0*p4}
                        solutions_pp.append(a.run(se_hb2('UZ1'), IPS=-2, NMX=8000, DS=dt, U=U))


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

        # plot the bifurcation y
        hbs, lps = s('HB'), s('LP')
        for l in lps:
            plt.scatter(l['PAR(1)'], l['U(' + str(uv) + ')'], s=200, marker='v', c='#5D6D7E')
        for h in hbs:
            plt.scatter(h['PAR(1)'], h['U(' + str(uv) + ')'], s=200, marker='o', c='#148F77')

    ax.set_xlim(-12.0, 0.0)
    ax.set_xlabel('eta')
    ax.set_ylabel('r')
    ax.set_title('One-parameter continuation')

if bifurcation_analysis and fixed_point_analysis:

    uvs = [1]
    uv_names = ['r', 'v', 'e', 'a']
    xlims = [(-6.5, -2.5), (), (), ()]
    ylims = [(0.0, 2.5), (), (), ()]
    for uv, name, xl, yl in zip(uvs, uv_names, xlims, ylims):

        # plot the principal continuation
        results, stability = get_cont_results(se, data_idx=[0, uv+1])
        col = get_line_collection(results[:, 1], results[:, 0], stability, c=['#76448A', '#5D6D7E'], linethickness=2.0)
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.add_collection(col)
        ax.autoscale()

        # plot the bifurcation y
        hbs, lps, shs, pds = se('HB'), se('LP'), se_hb2('LP'), se_hb2('PD')
        for l in lps:
            plt.scatter(l['PAR(1)'], l['U(' + str(uv) + ')'], s=300, marker='v', c='#5D6D7E')
        for h in hbs:
            plt.scatter(h['PAR(1)'], h['U(' + str(uv) + ')'], s=300, marker='o', c='#148F77')
        for s in shs:
            y = s['U(' + str(uv) + ')']
            plt.scatter(s['PAR(1)'], np.max(y), s=300, marker='p', c='#5D6D7E')
            plt.scatter(s['PAR(1)'], np.min(y), s=300, marker='p', c='#5D6D7E')
        for p in pds:
            y = p['U(' + str(uv) + ')']
            plt.scatter(p['PAR(1)'], np.max(y), s=300, marker='*', c='#5D6D7E')
            plt.scatter(p['PAR(1)'], np.min(y), s=300, marker='*', c='#5D6D7E')

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
        ax.set_xlim(xl[0], xl[1])
        ax.set_ylim(yl[0], yl[1])
        ax.set_xlabel('eta')
        ax.set_ylabel(name)
        ax.set_title('One-parameter continuation')

        # shade the area between the min/max of the limit cycle
        #ax.fill_between(x=results_hb2[:, 0], y1=p_min_hb2, y2=p_max_hb2, where=stability_hb2 < 0., alpha=0.3)
        #ax.fill_between(x=results_hb2[:, 0], y1=p_min_hb2, y2=p_max_hb2, where=stability_hb2 < 0., alpha=0.3)

# visualization of eta-alpha continuation
#########################################

if limit_cycle_analysis:

    results_lcb1, stability_lcb1 = get_cont_results(se_lcb1, data_idx=[0, -1])
    results_lcb2, stability_lcb2 = get_cont_results(se_lcb2, data_idx=[0, -1])
    col_lcb1 = get_line_collection(results_lcb1[:, 1], results_lcb1[:, 0], stability_lcb1, c='k',
                                   linestyles=['-', '--'], linethickness=4.0)
    col_lcb2 = get_line_collection(results_lcb2[:, 1], results_lcb2[:, 0], stability_lcb2, c='k',
                                   linestyles=['-', '--'], linethickness=4.0)

    results_lcs1, stability_lcs1, solution_lcs1, points_lcs1 = get_cont_results(se_lcs_upper, data_idx=[0, -1],
                                                                                extract_solution=[1, 2, 3, 11])
    results_lcs2, stability_lcs2, solution_lcs2, points_lcs2 = get_cont_results(se_lcs_lower, data_idx=[0, -1],
                                                                                extract_solution=[1, 2, 3, 11])
    col_lcs1 = get_line_collection(results_lcs1[:, 1], results_lcs1[:, 0], stability_lcs1, c='#148F77',
                                   linestyles=['-', '-'], linethickness=4.0)
    col_lcs2 = get_line_collection(results_lcs2[:, 1], results_lcs2[:, 0], stability_lcs2, c='#148F77',
                                   linestyles=['-', '-'], linethickness=4.0)

    results_fp1, stability_fp1 = get_cont_results(se_fp1, data_idx=[0, -1])
    results_fp2, stability_fp2 = get_cont_results(se_fp2, data_idx=[0, -1])
    col_fp1 = get_line_collection(results_fp1[:, 0], results_fp1[:, 1], stability_fp1, c='#5D6D7E', linestyles=['-', '--'],
                                  linethickness=4.0)
    col_fp2 = get_line_collection(results_fp2[:, 0], results_fp2[:, 1], stability_fp2, c='#5D6D7E', linestyles=['-', '--'],
                                  linethickness=4.0)

    fig2, ax2 = plt.subplots(nrows=2, figsize=(7, 16))

    # plot the lines
    ax2[0].add_collection(col_lcb1)
    ax2[0].add_collection(col_lcb2)
    ax2[0].add_collection(col_lcs1)
    ax2[0].add_collection(col_lcs2)
    ax2[0].add_collection(col_fp1)
    ax2[0].add_collection(col_fp2)

    # plot the bifurcation y
    plt.sca(ax2[0])
    bts, ghs, cps = se_lcb1('BT') + se_lcb2('BT'), se_lcb1('GH') + se_lcb2('GH'), se_fp1('CP') + se_fp2('CP')
    for b in bts:
        plt.scatter(b['PAR(1)'], b['PAR(3)'], s=300, marker='s', c='k')
    for g in ghs:
        plt.scatter(g['PAR(1)'], g['PAR(3)'], s=300, marker='o', c='#148F77')
    for c in cps:
        plt.scatter(c['PAR(1)'], c['PAR(3)'], s=300, marker='d', c='#5D6D7E')
    ax2[0].autoscale()
    ax2[0].set_xlim(-6.6, -2.9)
    ax2[0].set_ylim(0., 0.17)
    ax2[0].set_xlabel('eta')
    ax2[0].set_ylabel('alpha')
    ax2[0].set_title('Two-parameter continuation')

    # visualization of limit cycle period
    #####################################

    plt.sca(ax2[1])
    im = plt.imshow(period_solutions*0.01, aspect=1.0, interpolation='nearest', cmap='magma')
    divider = make_axes_locatable(ax2[1])
    cax = divider.append_axes("right", size="8%", pad=0.2)
    plt.colorbar(im, cax=cax, label='ms')
    ax2[1].set_xlabel('eta')
    ax2[1].set_ylabel('alpha')
    ax2[1].set_xticks(np.arange(period_solutions.shape[1])[::4])
    ax2[1].set_yticks(np.arange(period_solutions.shape[0])[::4])
    ax2[1].set_xticklabels(np.round(etas[::4], decimals=2))
    ax2[1].set_yticklabels(np.round(alphas[::4], decimals=2))
    ax2[1].set_title('Limit cycle period')
    plt.tight_layout()
    plt.savefig('Fig4_tmp.svg')

    # same shit for alpha-tau
    #########################

    results_lcb3, stability_lcb3 = get_cont_results(se_lcb3, data_idx=[0, -1])
    results_lcb4, stability_lcb4 = get_cont_results(se_lcb4, data_idx=[0, -1])
    col_lcb3 = get_line_collection(results_lcb3[:, 1], results_lcb3[:, 0], stability_lcb3, c='k',
                                   linestyles=['-', '--'], linethickness=4.0)
    col_lcb4 = get_line_collection(results_lcb4[:, 1], results_lcb4[:, 0], stability_lcb4, c='k',
                                   linestyles=['-', '--'], linethickness=4.0)

    results_lcs3, stability_lcs3, solution_lcs3, points_lcs3 = get_cont_results(se_lcs_upper2, data_idx=[0, -1],
                                                                                extract_solution=[1, 2, 3, 11])
    results_lcs4, stability_lcs4, solution_lcs4, points_lcs4 = get_cont_results(se_lcs_lower2, data_idx=[0, -1],
                                                                                extract_solution=[1, 2, 3, 11])
    col_lcs3 = get_line_collection(results_lcs3[:, 1], results_lcs3[:, 0], stability_lcs3, c='#148F77',
                                   linestyles=['-', '-'], linethickness=4.0)
    col_lcs4 = get_line_collection(results_lcs4[:, 1], results_lcs4[:, 0], stability_lcs4, c='#148F77',
                                   linestyles=['-', '-'], linethickness=4.0)

    results_fp3, stability_fp3 = get_cont_results(se_fp3, data_idx=[0, -1])
    results_fp4, stability_fp4 = get_cont_results(se_fp4, data_idx=[0, -1])
    col_fp3 = get_line_collection(results_fp3[:, 1], results_fp3[:, 0], stability_fp3, c='#5D6D7E',
                                  linestyles=['-', '--'], linethickness=4.0)
    col_fp4 = get_line_collection(results_fp4[:, 1], results_fp4[:, 0], stability_fp4, c='#5D6D7E',
                                  linestyles=['-', '--'], linethickness=4.0)

    fig4, ax4 = plt.subplots(figsize=(15, 8))

    # plot the lines
    ax4.add_collection(col_lcb3)
    ax4.add_collection(col_lcb4)
    ax4.add_collection(col_lcs3)
    ax4.add_collection(col_lcs4)
    ax4.add_collection(col_fp3)
    ax4.add_collection(col_fp4)

    # plot the bifurcation y
    bts, ghs, cps = se_lcb3('BT') + se_lcb4('BT'), se_lcb3('GH') + se_lcb4('GH'), se_fp3('CP') + se_fp4('CP')
    for b in bts:
        plt.scatter(b['PAR(1)'], b['PAR(4)'], s=300, marker='s', c='k')
    for g in ghs:
        plt.scatter(g['PAR(1)'], g['PAR(4)'], s=300, marker='o', c='#148F77')
    for c in cps:
        plt.scatter(c['PAR(1)'], c['PAR(4)'], s=300, marker='d', c='#5D6D7E')
    ax4.autoscale()
    ax4.set_xlim(-12.0, -3.0)
    ax4.set_ylim(0.0, 33.0)
    ax4.set_xlabel('eta')
    ax4.set_ylabel('tau')

    # visualization of limit cycle period
    #####################################

    fig5, ax5 = plt.subplots(figsize=(15, 10))
    plt.imshow(period_solutions2, aspect=1.0, interpolation='nearest', cmap='magma')
    plt.colorbar()
    ax5.set_xlabel('eta')
    ax5.set_ylabel('tau')
    ax5.set_xticks(np.arange(period_solutions2.shape[1])[::4])
    ax5.set_yticks(np.arange(period_solutions2.shape[0])[::4])
    ax5.set_xticklabels(etas[::4])
    ax5.set_yticklabels(taus[::4])

if phase_portrait_analysis:

    # visualization of phase portrait
    #################################

    _, _, _, lc_is = extract_from_solution(se_hb2('UZ1'), params=[], vars=[], time_vars=['U(1)', 'U(2)', 'U(3)', 'U(4)'])

    # 3d plot of pp for eta, alpha and e
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111, projection='3d')
    for spp in solutions_pp:
        _, results, _, _ = extract_from_solution(spp, params=[], vars=['MAX U(1)', 'MAX U(2)', 'MAX U(3)', 'MAX U(4)'])
        cutoff = int(0.6 * len(results[0]))
        val_range = max(results[0][cutoff:]) - min(results[0][cutoff:])
        if val_range > 1.5:
            ax6.plot(xs=results[1], ys=results[0], zs=results[2], linestyle=':', c='#48C9B0', linewidth=2.0,
                     alpha=0.7)
            ax6.plot(xs=results[1][cutoff:], ys=results[0][cutoff:], zs=results[2][cutoff:], linestyle='-', c='#148F77',
                     linewidth=6.0)
        else:
            ax6.plot(xs=results[1], ys=results[0], zs=results[2], linestyle=':', c='#AF7AC5', linewidth=2.0,
                     alpha=0.7)
            ax6.plot(xs=results[1][cutoff:], ys=results[0][cutoff:], zs=results[2][cutoff:], marker='o', c='#76448A',
                     markersize=20.0)

    ax6.plot(xs=lc_is[1], ys=lc_is[0], zs=lc_is[2], linestyle='--', c='k', linewidth=6.0)
    ax6.tick_params(axis='both', which='major', pad=20)
    ax6.set_xlabel('v', labelpad=30)
    ax6.set_ylabel('r', labelpad=30)
    ax6.set_zlabel('e', labelpad=30)

    # 3d plot for pp of eta, alpha and a
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(111, projection='3d')
    for spp in solutions_pp:
        _, results, _, _ = extract_from_solution(spp, params=[], vars=['MAX U(1)', 'MAX U(2)', 'MAX U(3)', 'MAX U(4)'])
        cutoff = int(0.6 * len(results[0]))
        val_range = max(results[0][cutoff:]) - min(results[0][cutoff:])
        if val_range > 1.5:
            ax7.plot(xs=results[1], ys=results[0], zs=results[3], linestyle=':', c='#48C9B0', linewidth=2.0,
                     alpha=0.7)
            ax7.plot(xs=results[1][cutoff:], ys=results[0][cutoff:], zs=results[3][cutoff:], linestyle='-',
                     c='#148F77', linewidth=6.0)
        else:
            ax7.plot(xs=results[1], ys=results[0], zs=results[3], linestyle=':', c='#AF7AC5', linewidth=2.0, alpha=0.7)
            ax7.plot(xs=results[1][cutoff:], ys=results[0][cutoff:], zs=results[3][cutoff:], marker='o', c='#76448A',
                     markersize=20.0)

    ax7.plot(xs=lc_is[1], ys=lc_is[0], zs=lc_is[3], linestyle='--', c='k', linewidth=4.0)
    ax7.tick_params(axis='both', which='major', pad=20)
    ax7.set_xlabel('v', labelpad=30)
    ax7.set_ylabel('r', labelpad=30)
    ax7.set_zlabel('a', labelpad=30)

plt.show()
