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
        self._first_run = True

        self._bifurcation_styles = {'LP': {'marker': 'v', 'color' : '#5D6D7E'},
                                    'HB': {'marker': 'o', 'color': '#148F77'},
                                    'CP': {'marker': 'd', 'color': '#5D6D7E'},
                                    'PD': {'marker': 'h', 'color': '#5D6D7E'},
                                    'BT': {'marker': 's', 'color': 'k'},
                                    'GH': {'marker': 'o', 'color': '#148F77'}
                                    }

    def run(self, vars, params, stability=True, period=False, timeseries=False, lyapunov_exp=False,
            starting_point=None, **auto_kwargs):

        # auto call
        ###########

        # extract starting point of continuation
        if 'IRS' in auto_kwargs or 's' in auto_kwargs:
            raise ValueError('Usage of keyword arguments `IRS` and `s` is disabled in PyAuto. To start from a previous'
                             'solution, use the `starting_point` keyword argument and provide a tuple of branch '
                             'number and point number as returned by the `run` method.')
        if not starting_point and not self._first_run:
            raise ValueError('A starting point is required for further continuation. Either provide a solution to '
                             'start from via the `starting_point` keyword argument or create a fresh PyAuto instance.')
        self._first_run = False

        # call to auto
        if starting_point:
            s = self.get_solution(*starting_point)
            solution = a.run(s, **auto_kwargs)
        else:
            solution = a.run(**auto_kwargs)

        # extract information from auto solution
        ########################################

        # extract branch and solution info
        branch = self.get_branch(solution)
        points = self.get_solution_keys(solution)
        self.auto_solutions[branch] = solution

        # extract continuation results
        summary = {}
        for i, point in enumerate(points):

            summary[point] = {}

            # get solution
            s = self.get_solution(branch, i+1)

            # extract vars and params from solution
            var_vals = self.get_vars(s, vars, timeseries)
            param_vals = self.get_params(s, params)

            # store solution information in summary
            summary[point]['bifurcation'] = self.get_solution_type(s)
            for var, val in zip(vars, var_vals):
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

        self.branches[branch] = summary.copy()
        return branch, summary

    def get_solution(self, branch, point):
        return self.auto_solutions[branch](point)

    def extract(self, keys, branch):
        return {key: np.asarray([val[key]for point, val in self.branches[branch].items()]) for key in keys}

    def plot_codim1(self, param, var, branch, ax=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        # extract information from branch solutions
        results = self.extract([param, var, 'stability', 'bifurcation'], branch)

        # plot bifurcation points
        bifurcation_point_kwargs = ['default_color', 'default_marker', 'default_size', 'custom_bf_styles']
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

    def plot_codim2(self, p1, p2):

        pass

    def plot_trajectory(self, v1, v2, v3=None):

        pass

    def plot_timeseries(self, v):

        pass

    def plot_bifurcation_points(self, solution_types, x_vals, y_vals, ax, default_color='k', default_marker='*',
                                default_size=50, custom_bf_styles=None):

        # set bifurcation styles
        bf_styles = self._bifurcation_styles.copy()
        if custom_bf_styles:
            bf_styles.update(custom_bf_styles)
        plt.sca(ax)

        # draw bifurcation points
        for bf, x, y in zip(solution_types, x_vals, y_vals):
            if bf not in "EPMXRG":
                if bf in bf_styles:
                    m = bf_styles[bf]['marker']
                    c = bf_styles[bf]['color']
                else:
                    m = default_marker
                    c = default_color
                plt.scatter(x, y, s=default_size, marker=m, c=c)
        return ax

    @staticmethod
    def get_solution_type(solution):
        return solution.b['solution'].b['TY name']

    @staticmethod
    def get_stability(s):
        return s.b['solution'].b['PT'] < 0

    @staticmethod
    def get_solution_keys(solution):
        return {s['PT']: {} for s in solution()}

    @staticmethod
    def get_branch(solution):
        return solution.data[0].BR

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
            branch_str = ' ' + str(branch) + ' '
            point_str = ' ' + str(point) + ' '

            for diag_tmp in diag_split:

                if branch_str in diag_tmp[:5] and point_str in diag_tmp[5:11] and \
                        ('Eigenvalue' in diag_tmp or 'Multiplier' in diag_tmp):

                    lyapunovs = []
                    i = 0
                    while True:

                        i += 1

                        # check whether solution is periodic or not
                        if 'Eigenvalue' in diag_tmp:
                            start_str = 'Eigenvalue  ' + str(i + 1) + ':  '
                            stop_str = '\n'
                            period = 0
                        else:
                            start_str = 'Multiplier  ' + str(i + 1) + '   '
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
