import os
import auto as a
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection


class PyAuto:

    def __init__(self, auto_dir=None):

        self.auto_solutions = {}
        self.branches = {}

        if auto_dir:
            os.chdir(auto_dir)
        self._dir = os.getcwd()
        self._last_cont = 0
        self._last_branch = 0

        self._bifurcation_styles = {'LP': {'marker': 'v', 'color' : '#5D6D7E'},
                                    'HB': {'marker': 'o', 'color': '#148F77'},
                                    'CP': {'marker': 'd', 'color': '#5D6D7E'},
                                    'PD': {'marker': 'h', 'color': '#5D6D7E'},
                                    'BT': {'marker': 's', 'color': 'k'},
                                    'GH': {'marker': 'o', 'color': '#148F77'}
                                    }

    def run(self, variables=None, params=None, extract_stability=True, extract_period=False, extract_timeseries=False,
            extract_lyapunov_exp=False, starting_point=None, starting_branch=None, starting_cont=None, **auto_kwargs):

        # auto call
        ###########

        # extract starting point of continuation
        if 'IRS' in auto_kwargs or 's' in auto_kwargs:
            raise ValueError('Usage of keyword arguments `IRS` and `s` is disabled in pyauto. To start from a previous'
                             'solution, use the `starting_point` keyword argument and provide a tuple of branch '
                             'number and point number as returned by the `run` method.')
        if not starting_point and self._last_cont != 0:
            raise ValueError('A starting point is required for further continuation. Either provide a solution to '
                             'start from via the `starting_point` keyword argument or create a fresh pyauto instance.')
        if not starting_cont:
            starting_cont = self._last_cont
        if not starting_branch:
            starting_branch = self._last_branch

        # call to auto
        if starting_point:
            _, s = self.get_solution(starting_branch, starting_cont, starting_point)
            solution = a.run(s, **auto_kwargs)
        else:
            solution = a.run(**auto_kwargs)

        # extract information from auto solution
        ########################################

        # extract branch and solution info
        branch, icp = self.get_branch_info(solution)
        points = self.get_solution_keys(solution)
        if branch in self.auto_solutions and icp in self.auto_solutions[branch]:
            solution = a.merge(solution, self.auto_solutions[branch])
        elif branch not in self.auto_solutions:
            self.auto_solutions[branch] = {}
            self.branches[branch] = {}
        self.auto_solutions[branch][icp] = solution
        self._last_branch = branch
        self._last_cont = icp

        # get all passed variables and params
        if variables is None:
            variables = self._get_all_var_keys(self.get_solution(branch, icp, list(points.keys())[0])[1])
        if params is None:
            params = self._get_all_param_keys(self.get_solution(branch, icp, list(points.keys())[0])[1])

        # extract continuation results
        summary = self._create_summary(solution=solution, branch=branch, icp=icp, points=points, variables=variables,
                                       params=params, timeseries=extract_timeseries, stability=extract_stability,
                                       period=extract_period, lyapunov_exp=extract_lyapunov_exp)

        self.branches[branch][icp] = summary.copy()
        return {'branch': branch, 'icp': icp, 'points': summary}

    def get_solution(self, branch, icp, point):
        if type(point) is str:
            s = self.auto_solutions[branch][icp](point)
            solution_name = point[:2]
        else:
            s = self.auto_solutions[branch][icp][0].labels.by_index[point-1]
            solution_name = list(s.keys())[0]
            s = s[solution_name]['solution']
        return solution_name, s

    def extract(self, keys, branch, icp):
        return {key: np.asarray([val[key]for point, val in self.branches[branch][icp].items()]) for key in keys}

    def plot_continuation(self, param, var, branch, icp, ax=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        # extract information from branch solutions
        results = self.extract([param, var, 'stability', 'bifurcation'], branch, icp)

        # plot bifurcation points
        bifurcation_point_kwargs = ['default_color', 'default_marker', 'default_size', 'custom_bf_styles',
                                    'ignore']
        kwargs_tmp = {key: kwargs.pop(key) for key in bifurcation_point_kwargs if key in kwargs}
        ax = self.plot_bifurcation_points(solution_types=results['bifurcation'], x_vals=results[param],
                                          y_vals=results[var], ax=ax, **kwargs_tmp)

        # plot main continuation
        line_col = self.get_line_collection(x=results[param], y=results[var], stability=results['stability'], **kwargs)
        ax.add_collection(line_col)
        ax.autoscale()

        # cosmetics
        ax.set_xlabel(param)
        ax.set_ylabel(var)

        return ax

    def plot_trajectory(self, v1, v2, v3=None):

        pass

    def plot_timeseries(self, v):

        pass

    def plot_bifurcation_points(self, solution_types, x_vals, y_vals, ax, default_color='k', default_marker='*',
                                default_size=50, ignore=None, custom_bf_styles=None):

        if not ignore:
            ignore = []

        # set bifurcation styles
        bf_styles = self._bifurcation_styles.copy()
        if custom_bf_styles:
            bf_styles.update(custom_bf_styles)
        plt.sca(ax)

        # draw bifurcation points
        for bf, x, y in zip(solution_types, x_vals, y_vals):
            if bf not in "EPMXRG" and bf not in ignore:
                if bf in bf_styles:
                    m = bf_styles[bf]['marker']
                    c = bf_styles[bf]['color']
                else:
                    m = default_marker
                    c = default_color
                plt.scatter(x, y, s=default_size, marker=m, c=c)
        return ax

    def _create_summary(self, solution, branch, icp, points, variables, params, timeseries, stability, period,
                        lyapunov_exp):

        summary = {}
        for point in points:

            summary[point] = {}

            # get solution
            solution_type, s = self.get_solution(branch, icp, point)

            # extract variables and params from solution
            var_vals = self.get_vars(s, variables, timeseries)
            param_vals = self.get_params(s, params)

            # store solution information in summary
            summary[point]['bifurcation'] = solution_type
            for var, val in zip(variables, var_vals):
                summary[point][var] = val
            for param, val in zip(params, param_vals):
                summary[point][param] = val
            if stability:
                summary[point]['stability'] = self.get_stability(s)
            if period:
                summary[point]['period'] = summary[point]['PAR(11)'] if 'PAR(11)' in params else \
                    self.get_params(s, ['PAR(11)'])[0]
            if lyapunov_exp:
                summary[point]['lyapunov_exponents'] = self.get_lyapunov_exponent(solution, branch, point)

        return summary

    @staticmethod
    def get_stability(s):
        return s.b['solution'].b['PT'] < 0

    @staticmethod
    def get_solution_keys(solution):
        return {s['PT']: {} for s in solution()}

    @staticmethod
    def get_branch_info(solution):
        return solution[0].BR, tuple(solution[0].c['ICP'])

    @staticmethod
    def get_vars(solution, vars, extract_timeseries=False):
        if hasattr(solution, 'b') and extract_timeseries:
            solution = solution.b['solution']
        return [solution[v] for v in vars]

    @staticmethod
    def get_params(solution, params):
        if hasattr(solution, 'b'):
            solution = solution.b['solution']
        return [solution[p] for p in params]

    @staticmethod
    def get_lyapunov_exponent(solution, branch, point):

        diag = solution[0].diagnostics.data
        N = len(diag)

        # go through auto_solutions of diagnostic data
        for point_idx in range(N):

            # extract relevant diagnostic text output
            diag_split = diag[point_idx]['Text'].split('\n\n')

            # check whether branch and point identifiers match the targets
            branch_str = f' {str(branch)} '
            point_str = f' {str(point)} '

            for diag_tmp in diag_split:

                if branch_str in diag_tmp[:5] and point_str in diag_tmp[5:11] and \
                        ('Eigenvalue' in diag_tmp or 'Multiplier' in diag_tmp):

                    lyapunovs = []
                    i = 0
                    while True:

                        i += 1

                        # check whether solution is periodic or not
                        if 'Eigenvalue' in diag_tmp:
                            start_str = f'Eigenvalue  {str(i + 1)}:  '
                            stop_str = '\n'
                            period = 0
                        else:
                            start_str = f'Multiplier  str(i + 1)   '
                            stop_str = '  Abs. Val.'
                            period = float(diag_split[2].split(' ')[-1])

                        # extract eigenvalues/floquet multipliers
                        if start_str in diag_tmp:
                            start = diag_tmp.index(start_str) + len(start_str)
                            stop = diag_tmp[start:].index(stop_str) + start if stop_str in diag_tmp[start:] else None
                            diag_tmp_split = diag_tmp[start:stop].split(' ')
                            real = float(diag_tmp_split[1]) if diag_tmp_split[0] == ' ' else float(diag_tmp_split[0])
                            imag = float(diag_tmp_split[-1])
                        else:
                            break

                        # calculate lyapunov exponent
                        lyapunov = np.log(complex(real, imag)) / period if period else real
                        lyapunovs.append(lyapunov)

                    if lyapunovs:
                        return lyapunovs

        return []

    @staticmethod
    def _get_all_var_keys(solution):
        return [f'U({i+1})' for i in range(solution['NDIM'])]

    @staticmethod
    def _get_all_param_keys(solution):
        return solution.PAR.coordnames

    @staticmethod
    def get_line_collection(x, y, stability=None, line_style_stable='solid', line_style_unstable='dotted', **kwargs):
        """

        :param y:
        :param x:
        :param stability:
        :param line_style_stable
        :param line_style_unstable

        :return:
        """

        # combine y and param vals
        y = np.reshape(y, (y.shape[0], 1))
        x = np.reshape(x, (len(x), 1))
        y = np.append(x, y, axis=1)

        # if stability was passed, collect indices for stable line segments
        ###################################################################

        if stability is not None:

            # collect indices
            stability = np.asarray(stability, dtype='int')
            stability_changes = np.concatenate([np.zeros((1,)), np.diff(stability)])
            idx_changes = np.sort(np.argwhere(stability_changes != 0))
            idx_changes = np.append(idx_changes, len(stability_changes))

            # create line segments
            lines, styles = [], []
            idx_old = 1
            for idx in idx_changes:
                lines.append(y[idx_old-1:idx, :])
                styles.append(line_style_stable if stability[idx_old] else line_style_unstable)
                idx_old = idx

        else:

            lines = [y]
            styles = [line_style_stable]

        return LineCollection(segments=lines, linestyles=styles, **kwargs)
