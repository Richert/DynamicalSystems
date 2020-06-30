import os
import auto as a
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Union, Any
import pickle


class PyAuto:

    def __init__(self, auto_dir: str = None) -> None:

        # open attributes
        self.auto_solutions = {}
        self.results = {}

        # private attributes
        if auto_dir:
            os.chdir(auto_dir)
        self._dir = os.getcwd()
        self._last_cont = None
        self._cont_num = 0
        self._results_map = {}
        self._branches = {}
        self._bifurcation_styles = {'LP': {'marker': 'v', 'color' : '#5D6D7E'},
                                    'HB': {'marker': 'o', 'color': '#148F77'},
                                    'CP': {'marker': 'd', 'color': '#5D6D7E'},
                                    'PD': {'marker': 'h', 'color': '#5D6D7E'},
                                    'BT': {'marker': 's', 'color': 'k'},
                                    'GH': {'marker': 'o', 'color': '#148F77'}
                                    }

    def run(self, variables: list = None, params: list = None, get_stability: bool = True,
            get_period: bool = False, get_timeseries: bool = False, get_eigenvals: bool = False,
            get_lyapunov_exp: bool = False, starting_point: Union[str, int] = None, origin: dict = None,
            bidirectional: bool = False, name: str = None, **auto_kwargs) -> tuple:
        """Wraps auto-07p command `run` and stores requested solution details on instance.

        Parameters
        ----------
        variables
        params
        get_stability
        get_period
        get_timeseries
        get_eigenvals
        get_lyapunov_exp
        starting_point
        origin
        bidirectional
        name
        auto_kwargs

        Returns
        -------
        tuple
        """

        # auto call
        ###########

        # extract starting point of continuation
        if 'IRS' in auto_kwargs or 's' in auto_kwargs:
            raise ValueError('Usage of keyword arguments `IRS` and `s` is disabled in pyauto. To start from a previous'
                             'solution, use the `starting_point` keyword argument and provide a tuple of branch '
                             'number and point number as returned by the `run` method.')
        if not starting_point and self._last_cont:
            raise ValueError('A starting point is required for further continuation. Either provide a solution to '
                             'start from via the `starting_point` keyword argument or create a fresh pyauto instance.')
        if origin is None:
            origin = self._last_cont
        elif type(origin) is str:
            origin = self._results_map[origin]
        elif type(origin) is not int:
            origin = origin.pyauto_key

        # call to auto
        solution = self._call_auto(starting_point, origin, **auto_kwargs)

        # extract information from auto solution
        ########################################

        # extract branch and solution info
        new_branch, new_icp = self.get_branch_info(solution)
        new_points = self.get_solution_keys(solution)

        # get all passed variables and params
        _, solution_tmp = self.get_solution(point=new_points[0], cont=solution)
        if variables is None:
            variables = self._get_all_var_keys(solution_tmp)
        if params is None:
            params = self._get_all_param_keys(solution_tmp)

        # extract continuation results
        if new_icp[0] == 14:
            get_stability = False

        summary = self._create_summary(solution=solution, points=new_points, variables=variables,
                                       params=params, timeseries=get_timeseries, stability=get_stability,
                                       period=get_period, eigenvals=get_eigenvals, lyapunov_exp=get_lyapunov_exp)

        # store solution and extracted information in pyauto
        ####################################################

        # merge auto solutions if necessary and create key for auto solution
        if new_branch in self._branches and origin in self._branches[new_branch] \
                and new_icp in self._branches[new_branch][origin]:

            # get key from old solution and merge with new solution
            solution_old = self.get_solution(origin)
            pyauto_key = solution_old.pyauto_key
            solution, summary = self.merge(pyauto_key, solution, summary, new_icp)

        elif name == 'bidirect:cont2' and not bidirectional and 'DS' in auto_kwargs and auto_kwargs['DS'] == '-':

            # get key from old solution and merge with new solution
            solution_old = self._last_cont
            pyauto_key = solution_old.pyauto_key
            solution, summary = self.merge(pyauto_key, solution, summary, new_icp)

        else:

            # create pyauto key for solution
            pyauto_key = self._cont_num + 1 if self._cont_num in self.auto_solutions else self._cont_num
            solution.pyauto_key = pyauto_key

        # set up dictionary fields in _branches for new solution
        if new_branch not in self._branches:
            self._branches[new_branch] = {pyauto_key: []}
        elif pyauto_key not in self._branches[new_branch]:
            self._branches[new_branch][pyauto_key] = []

        # store auto solution under unique pyauto cont
        self.auto_solutions[pyauto_key] = solution
        self._last_cont = solution
        self._branches[new_branch][pyauto_key].append(new_icp)

        self.results[pyauto_key] = summary.copy()
        self._cont_num = pyauto_key
        if name and name != 'bidirect:cont2':
            self._results_map[name] = pyauto_key

        # if continuation should be bidirectional, call this method again with reversed continuation direction
        ######################################################################################################

        if bidirectional:
            auto_kwargs.pop('DS', None)
            summary, solution = self.run(variables=variables, params=params, get_stability=get_stability,
                                         get_period=get_period, get_timeseries=get_timeseries,
                                         get_eigenvals=get_eigenvals, get_lyapunov_exp=get_lyapunov_exp,
                                         starting_point=starting_point, origin=origin, bidirectional=False,
                                         name='bidirect:cont2', DS='-', **auto_kwargs)

        return summary.copy(), solution

    def merge(self, key: int, cont, summary: dict, icp: tuple):
        """Merges two solutions from two separate auto continuations.

        Parameters
        ----------
        key
            PyAuto identifier under which the merged solution should be stored. Must be equal to identifier of first
            continuation.
        cont
            auto continuation object that should be merged with the continuation object under `key`.
        summary
            PyAuto continuation summary that should be merged with continuation summary under `key`.
        icp
            Continuation parameter that was used in both continuations that are to be merged.
        """

        # merge solutions
        #################

        # call merge in auto
        solution = a.merge(self.auto_solutions[key] + cont)
        solution.pyauto_key = key

        # store solution in pyauto
        self.auto_solutions[key] = solution
        self._last_cont = solution

        # merge results summaries
        #########################

        summary_old = self.results[key]

        # extract end points and icp values at end points
        points_old, points_new = list(summary_old), list(summary)
        start_old, start_new, end_old, end_new = points_old[0], points_new[0], points_old[-1], points_new[-1]

        # connect starting points of both continuations and re-label points accordingly
        conts_sorted = [summary, summary_old]
        new_keys = [points_new[::-1], points_old]
        old_keys = [points_new, points_old]
        end_point = end_new

        # move points into combined summary
        summary_final = {}
        for p1, p2 in zip(new_keys[0], old_keys[0]):
            summary_final[p1] = conts_sorted[0][p2]
        for p1, p2 in zip(new_keys[1], old_keys[1]):
            summary_final[p1+end_point] = conts_sorted[1][p2]

        # store updated summary
        self.results[key] = summary_final

        return solution, summary_final

    def get_summary(self, cont: Union[Any, str, int], point=None) -> dict:
        """Extract summary of continuation from PyAuto.

        Parameters
        ----------
        cont
        point

        Returns
        -------
        dict

        """

        # get continuation summary
        if type(cont) is int:
            summary = self.results[cont]
        elif type(cont) is str:
            summary = self.results[self._results_map[cont]]
        else:
            summary = self.results[cont.pyauto_key]

        # return continuation or point summary
        if not point:
            return summary
        elif type(point) is str:
            n = int(point[2:]) if len(point) > 2 else 1
            i = 1
            for p, p_info in summary.items():
                if point[:2] == p_info['bifurcation']:
                    if i == n:
                        return summary[p]
                    i += 1
            else:
                raise KeyError(f'Invalid point: {point} was not found on continuation {cont}.')

        return summary[point]

    def get_solution(self, cont: Union[Any, str, int], point: Union[str, int] = None) -> Union[Any, tuple]:
        """

        Parameters
        ----------
        cont
        point

        Returns
        -------

        """

        # extract continuation object
        if type(cont) is int:
            cont = self.auto_solutions[cont]
        elif type(cont) is str:
            cont = self.auto_solutions[self._results_map[cont]]
        branch, icp = self.get_branch_info(cont)

        if point is None:
            return cont

        # extract solution point from continuation object and its solution type
        if type(point) is str:
            s = cont(point)
            solution_name = point[:2]
        else:
            for idx in range(len(cont.data)):
                s = cont[idx].labels.by_index[point]
                if s:
                    solution_name = list(s.keys())[0]
                    break
            else:
                raise ValueError(f'Invalid point {point} for continuation with ICP={icp} on branch {branch}.')
            if solution_name != 'No Label':
                s = s[solution_name]['solution']

        return solution_name, s

    def extract(self, keys: list, cont: Union[Any, str, int], point: Union[str, int] = None) -> dict:
        summary = self.get_summary(cont, point=point)
        if point:
            return {key: np.asarray(summary[key]) for key in keys}
        points = np.sort(list(summary))
        return {key: np.asarray([summary[p][key] for p in points]) for key in keys}

    def to_file(self, filename: str, include_auto_results: bool = False, **kwargs) -> None:
        """Save continuation results on disc.

        Parameters
        ----------
        filename
        include_auto_results

        Returns
        -------
        None
        """

        data = {'results': self.results, '_branches': self._branches, '_results_map': self._results_map}
        if include_auto_results:
            data['auto_solutions'] = self.auto_solutions
        data.update(kwargs)

        for key in kwargs:
            if hasattr(self, key):
                print(f'WARNING: {key} is an attribute of PyAuto instances. To be able to build a new instance of '
                      f'PyAuto via the `from_file` method from this file, you need to provide a different attribute '
                      f'name.')

        try:
            pickle.dump(data, open(filename, 'x'))
        except (FileExistsError, TypeError):
            pickle.dump(data, open(filename, 'wb'))

    def plot_continuation(self, param: str, var: str, cont: Union[Any, str, int], ax: plt.Axes = None,
                          force_axis_lim_update: bool = False, **kwargs) -> plt.Axes:
        """Line plot of 1D/2D parameter continuation and the respective codimension 1/2 bifurcations.

        Parameters
        ----------
        param
        var
        cont
        ax
        force_axis_lim_update
        kwargs

        Returns
        -------
        plt.Axes
        """

        if ax is None:
            fig, ax = plt.subplots()
        label_pad = kwargs.pop('labelpad', 5)
        tick_pad = kwargs.pop('tickpad', 5)
        axislim_pad = kwargs.pop('axislimpad', 0)

        # extract information from branch solutions
        results = self.extract([param, var, 'stability', 'bifurcation'], cont=cont)

        # plot bifurcation points
        bifurcation_point_kwargs = ['default_color', 'default_marker', 'default_size', 'custom_bf_styles',
                                    'ignore']
        kwargs_tmp = {key: kwargs.pop(key) for key in bifurcation_point_kwargs if key in kwargs}
        ax = self.plot_bifurcation_points(solution_types=results['bifurcation'], x_vals=results[param],
                                          y_vals=results[var], ax=ax, **kwargs_tmp)

        # set title variable if passed
        tvar = kwargs.pop('title_var', None)
        if tvar:
            tvar_results = self.extract([tvar], cont=cont)
            tval = tvar_results[tvar][0]
            ax.set_title(f"{tvar} = {tval}")

        # plot main continuation
        x, y = results[param], results[var]
        line_col = self._get_line_collection(x=x, y=y, stability=results['stability'], **kwargs)
        ax.add_collection(line_col)
        ax.autoscale()

        # cosmetics
        ax.tick_params(axis='both', which='major', pad=tick_pad)
        ax.set_xlabel(param, labelpad=label_pad)
        ax.set_ylabel(var, labelpad=label_pad)
        self._update_axis_lims(ax, ax_data=[x, y], padding=axislim_pad, force_update=force_axis_lim_update)

        return ax

    def plot_trajectory(self, vars: Union[list, tuple], cont: Union[Any, str, int], point: Union[str, int] = None,
                        ax: plt.Axes = None, force_axis_lim_update: bool = False, **kwargs) -> plt.Axes:
        """Plot trajectory of state variables through phase space over time.

        Parameters
        ----------
        vars
        cont
        point
        ax
        force_axis_lim_update
        kwargs

        Returns
        -------
        plt.Axes
        """

        # extract information from branch solutions
        results = self.extract(list(vars) + ['stability'], cont=cont, point=point)

        if len(vars) == 2:

            # create 2D plot
            if ax is None:
                fig, ax = plt.subplots()

            # plot phase trajectory
            line_col = self._get_line_collection(x=results[vars[0]], y=results[vars[1]], stability=results['stability'],
                                                 **kwargs)
            ax.add_collection(line_col)
            ax.autoscale()

            # cosmetics
            ax.set_xlabel(vars[0])
            ax.set_ylabel(vars[1])

        elif len(vars) == 3:

            # create 3D plot
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            label_pad = kwargs.pop('labelpad', 30)
            tick_pad = kwargs.pop('tickpad', 20)
            axislim_pad = kwargs.pop('axislimpad', 0.1)

            # plot phase trajectory
            x, y, z = results[vars[0]], results[vars[1]], results[vars[2]]
            line_col = self._get_3d_line_collection(x=x, y=y, z=z, stability=results['stability'], **kwargs)
            ax.add_collection3d(line_col)
            ax.autoscale()

            # cosmetics
            ax.tick_params(axis='both', which='major', pad=tick_pad)
            ax.set_xlabel('v', labelpad=label_pad)
            ax.set_ylabel('r', labelpad=label_pad)
            ax.set_zlabel('e', labelpad=label_pad)
            self._update_axis_lims(ax, [x, y, z], padding=axislim_pad, force_update=force_axis_lim_update)

        else:

            raise ValueError('Invalid number of state variables to plot. First argument can only take 2 or 3 state'
                             'variable names as input.')

        return ax

    def plot_timeseries(self, var: str, cont: Union[Any, str, int], points: list = None, ax: plt.Axes = None,
                        linespecs: list = None, **kwargs) -> plt.Axes:
        """

        Parameters
        ----------
        var
        cont
        points
        ax
        linespecs
        kwargs

        Returns
        -------

        """

        # extract information from branch solutions
        if not points:
            points = ['RG']
            points_tmp = self.results[self._results_map[cont] if type(cont) is str else cont].keys()
            results_tmp = [self.extract([var] + ['time'], cont=cont, point=p) for p in points_tmp]
            results = [{key: [] for key in results_tmp[0].keys()}]
            for r in results_tmp:
                for key in r:
                    results[0][key].append(np.squeeze(r[key]))
            for key in results[0].keys():
                results[0][key] = np.asarray(results[0][key]).squeeze()
        else:
            results = [self.extract([var] + ['time'], cont=cont, point=p) for p in points]

        # create plot
        if ax is None:
            fig, ax = plt.subplots()

        # plot phase trajectory
        if not linespecs:
            linespecs = [dict() for _ in range(len(points))]
        for i in range(len(points)):
            time = results[i]['time']
            kwargs_tmp = kwargs.copy()
            kwargs_tmp.update(linespecs[i])
            line_col = self._get_line_collection(x=time, y=results[i][var], **kwargs_tmp)
            ax.add_collection(line_col)
        ax.autoscale()
        ax.legend(points)

        return ax

    def plot_bifurcation_points(self, solution_types, x_vals, y_vals, ax, default_color='k', default_marker='*',
                                default_size=50, ignore=None, custom_bf_styles=None):
        """

        Parameters
        ----------
        solution_types
        x_vals
        y_vals
        ax
        default_color
        default_marker
        default_size
        ignore
        custom_bf_styles

        Returns
        -------

        """

        if not ignore:
            ignore = []

        # set bifurcation styles
        if custom_bf_styles:
            for key, args in custom_bf_styles.items():
                self.update_bifurcation_style(key, **args)
        bf_styles = self._bifurcation_styles.copy()
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
                if y.shape and np.sum(y.shape) > 1:
                    plt.scatter(x, y.max(), s=default_size, marker=m, c=c)
                    plt.scatter(x, y.min(), s=default_size, marker=m, c=c)
                else:
                    plt.scatter(x, y, s=default_size, marker=m, c=c)
        return ax

    def update_bifurcation_style(self, bf_type: str, marker: str = None, color: str = None):
        if bf_type in self._bifurcation_styles:
            if marker:
                self._bifurcation_styles[bf_type]['marker'] = marker
            if color:
                self._bifurcation_styles[bf_type]['color'] = color
        else:
            if marker is None:
                marker = 'o'
            if color is None:
                color = 'k'
            self._bifurcation_styles.update({bf_type: {'marker': marker, 'color': color}})

    def plot_heatmap(self, x: np.array, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        from seaborn import heatmap
        return heatmap(x, ax=ax, **kwargs)

    def _create_summary(self, solution: Union[Any, dict], points: list, variables: list, params: list,
                        timeseries: bool, stability: bool, period: bool, eigenvals: bool, lyapunov_exp: bool):
        """Creates summary of auto continuation and stores it in dictionary.

        Parameters
        ----------
        solution
        points
        variables
        params
        timeseries
        stability
        period
        eigenvals
        lyapunov_exp

        Returns
        -------

        """
        summary = {}
        for point in points:

            # get solution
            solution_type, s = self.get_solution(cont=solution, point=point)

            if solution_type != 'No Label' and solution_type != 'MX':

                summary[point] = {}

                # extract variables and params from solution
                var_vals = self.get_vars(s, variables, timeseries)
                param_vals = self.get_params(s, params)

                # store solution information in summary
                summary[point]['bifurcation'] = solution_type
                branch, _ = self.get_branch_info(s)
                for var, val in zip(variables, var_vals):
                    summary[point][var] = val
                if len(var_vals) > len(variables) and timeseries:
                    summary[point]['time'] = var_vals[-1]
                for param, val in zip(params, param_vals):
                    summary[point][param] = val
                if stability:
                    summary[point]['stability'] = self.get_stability(solution, s, point)
                if period or lyapunov_exp or eigenvals:
                    p = summary[point]['PAR(11)'] if 'PAR(11)' in params else self.get_params(s, ['PAR(11)'])[0]
                    if period:
                        summary[point]['period'] = p
                    if eigenvals or lyapunov_exp:
                        evs = self.get_eigenvalues(solution, branch, point)
                        if eigenvals:
                            summary[point]['eigenvalues'] = evs
                        if lyapunov_exp:
                            summary[point]['lyapunov_exponents'] = self.get_lyapunov_exponent(evs, p)

        return summary

    def _call_auto(self, starting_point: Union[str, int], origin: Union[Any, dict], **auto_kwargs) -> Any:
        if starting_point:
            _, s = self.get_solution(point=starting_point, cont=origin)
            solution = a.run(s, **auto_kwargs)
        else:
            solution = a.run(**auto_kwargs)
        return self._start_from_solution(solution)

    def _update_axis_lims(self, ax: Union[plt.Axes, Axes3D], ax_data: list, padding: float = 0.,
                          force_update: bool = False) -> None:
        ax_names = ['x', 'y', 'z']
        for i, data in enumerate(ax_data):
            axis_limits = self._get_axis_lims(np.asarray(data), padding=padding)
            if force_update:
                min_val, max_val = axis_limits
            else:
                min_val, max_val = eval(f"ax.get_{ax_names[i]}lim()")
                min_val, max_val = np.min([min_val, axis_limits[0]]), np.max([max_val, axis_limits[1]])
            eval(f"ax.set_{ax_names[i]}lim(min_val, max_val)")

    @classmethod
    def from_file(cls, filename: str, auto_dir: str = None) -> Any:
        """

        Parameters
        ----------
        filename
        auto_dir

        Returns
        -------
        Any
        """
        pyauto_instance = cls(auto_dir)
        data = pickle.load(open(filename, 'rb'))
        for key, val in data.items():
            if hasattr(pyauto_instance, key):
                attr = getattr(pyauto_instance, key)
                if type(attr) is dict:
                    attr.update(val)
                else:
                    raise AttributeError(f'Attribute {key} is already contained on this PyAuto instance and cannot be '
                                         f'set.')
            else:
                setattr(pyauto_instance, key, val)
        return pyauto_instance

    @staticmethod
    def get_stability(solution, s, point) -> bool:

        point_idx = get_point_idx(solution[0].diagnostics.data, point=point)
        diag = solution[0].diagnostics.data[point_idx]['Text']

        if "Eigenvalues" in diag:
            diag_line = "Eigenvalues  :   Stable:  "
        elif "Multipliers" in diag:
            diag_line = "Multipliers:     Stable:  "
        else:
            return s.b['solution'].b['PT'] < 0
        idx = diag.find(diag_line) + len(diag_line)
        value = int(diag[idx:].split("\n")[0])
        target = s.data['NDIM']
        return value >= target

    @staticmethod
    def get_solution_keys(solution: Any) -> list:
        keys = []
        for idx in range(len(solution.data)):
            keys += [key for key, val in solution[idx].labels.by_index.items()
                     if val and 'solution' in tuple(val.values())[0]]
        return keys

    @staticmethod
    def get_branch_info(solution: Any) -> tuple:
        try:
            branch, icp = solution[0].BR, solution[0].c['ICP']
        except (AttributeError, ValueError):
            try:
                branch, icp = solution['BR'], solution.c['ICP']
            except AttributeError:
                icp = solution[0].c['ICP']
                i = 0
                while i < 10:
                    try:
                        sol_key = list(solution[0].labels.by_index.keys())[i]
                        branch = solution[0].labels.by_index[sol_key]['RG']['solution']['data']['BR']
                        break
                    except KeyError as e:
                        i += 1
                        if i == 10:
                            raise e
        icp = (icp,) if type(icp) is int else tuple(icp)
        return branch, icp

    @staticmethod
    def get_vars(solution: Any, vars: list, extract_timeseries: bool = False) -> list:
        if hasattr(solution, 'b') and extract_timeseries:
            solution = solution.b['solution']
            solutions = [solution.indepvararray]
        else:
            solutions = []
        solutions = [solution[v] for v in vars] + solutions
        return solutions

    @staticmethod
    def get_params(solution, params):
        if hasattr(solution, 'b'):
            solution = solution.b['solution']
        return [solution[p] for p in params]

    @staticmethod
    def get_eigenvalues(solution, branch: int, point: int) -> list:
        """extracts eigenvalue spectrum from a solution point on a branch of solution

        Parameters
        ----------
        solution
        branch
        point

        Returns
        -------
        list
            List of eigenvalues. If solution is periodic, the list contains floquet multipliers instead of eigenvalues.
        """

        eigenvals = []

        # extract point index from diagnostics
        point_idx = get_point_idx(solution[0].diagnostics, point=point)
        diag = solution[0].diagnostics.data[point_idx]['Text']
        diag_split = diag.split('\n')

        # find index of line in diagnostic output where eigenvalue information are starting
        idx = 0
        while idx < len(diag_split):
            if 'Stable:' in diag_split[idx]:
                break
            idx += 1

        # check whether branch and point identifiers match the targets
        branch_str = f' {branch} '
        point_str = f' {point+1} '

        if "NOTE:No converge" in diag:
            return eigenvals

        if branch_str in diag_split[idx] and point_str in diag_split[idx] and \
                ('Eigenvalue' in diag or 'Multiplier' in diag_split[idx]):

            # go through lines of system diagnostics and extract eigenvalues/floquet multipliers
            i = 1
            while i < len(diag_split) - idx:

                diag_tmp = diag_split[i + idx]
                diag_tmp_split = [d for d in diag_tmp.split(' ') if d != ''][2:]

                # check whether line contains eigenvals or floquet mults. If not, stop while loop.
                if not diag_tmp_split:
                    break
                if 'Eigenvalue' not in diag_tmp_split[0] and 'Multiplier' not in diag_tmp_split[0]:
                    break

                # extract eigenvalues/floquet multipliers
                idx2 = diag_tmp_split.index(f"{i}")
                real = float(diag_tmp_split[idx2+1])
                imag = float(diag_tmp_split[idx2+2])
                eigenvals.append(complex(real, imag))

                i += 1

        return eigenvals

    @staticmethod
    def get_lyapunov_exponent(eigenvals, period):
        return [np.real(np.log(ev)/period) if period else np.real(ev) for ev in eigenvals]

    @staticmethod
    def _get_all_var_keys(solution):
        return [f'U({i+1})' for i in range(solution['NDIM'])]

    @staticmethod
    def _get_all_param_keys(solution):
        return solution.PAR.coordnames

    def _start_from_solution(self, solution: Any) -> Any:
        diag = str(solution[0].diagnostics)
        if 'Starting direction of the free parameter(s)' in diag and len(self.get_solution_keys(solution)) == 1:
            solution = a.run(solution)
        return solution

    @staticmethod
    def _get_line_collection(x, y, stability=None, line_style_stable='solid', line_style_unstable='dotted',
                             line_color_stable='k', line_color_unstable='k', **kwargs) -> LineCollection:
        """

        Parameters
        ----------
        x
        y
        stability
        line_style_stable
        line_style_unstable
        line_color_stable
        line_color_unstable
        kwargs

        Returns
        -------
        LineCollection
        """

        # combine y and param vals
        x = np.reshape(x, (x.squeeze().shape[0], 1))
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_max = np.reshape(y.max(axis=1), (y.shape[0], 1))
            y_min = np.reshape(y.min(axis=1), (y.shape[0], 1))
            y_min = np.append(x, y_min, axis=1)
            y = y_max
            add_min = True
        else:
            y = np.reshape(y, (y.squeeze().shape[0], 1))
            add_min = False
        y = np.append(x, y, axis=1)

        # if stability was passed, collect indices for stable line segments
        ###################################################################

        if stability is not None and np.sum(stability.shape) > 1:

            # collect indices
            stability = np.asarray(stability, dtype='int')
            stability_changes = np.concatenate([np.zeros((1,)), np.diff(stability)])
            idx_changes = np.sort(np.argwhere(stability_changes != 0))
            idx_changes = np.append(idx_changes, len(stability_changes))

            # create line segments
            lines, styles, colors = [], [], []
            idx_old = 1
            for idx in idx_changes:
                lines.append(y[idx_old-1:idx, :])
                styles.append(line_style_stable if stability[idx_old] else line_style_unstable)
                colors.append(line_color_stable if stability[idx_old] else line_color_unstable)
                if add_min:
                    lines.append(y_min[idx_old - 1:idx, :])
                    styles.append(line_style_stable if stability[idx_old] else line_style_unstable)
                    colors.append(line_color_stable if stability[idx_old] else line_color_unstable)
                idx_old = idx

        else:

            lines = [y, y_min] if add_min else [y]
            styles = [line_style_stable, line_style_stable] if add_min else [line_style_stable]
            colors = [line_color_stable, line_color_stable] if add_min else [line_color_stable]

        colors = kwargs.pop('colors', colors)
        return LineCollection(segments=lines, linestyles=styles, colors=colors, **kwargs)

    @staticmethod
    def _get_3d_line_collection(x, y, z, stability=None, line_style_stable='solid', line_style_unstable='dotted',
                                **kwargs) -> Line3DCollection:
        """

        Parameters
        ----------
        x
        y
        z
        stability
        line_style_stable
        line_style_unstable
        kwargs

        Returns
        -------
        Line3DCollection
        """

        # combine y and param vals
        x = np.reshape(x, (x.squeeze().shape[0], 1))
        y = np.reshape(y, (y.squeeze().shape[0], 1))
        z = np.reshape(z, (z.squeeze().shape[0], 1))
        y = np.append(x, y, axis=1)
        y = np.append(y, z, axis=1)

        # if stability was passed, collect indices for stable line segments
        ###################################################################

        if stability is not None and np.sum(stability.shape) > 1:

            # collect indices
            stability = np.asarray(stability, dtype='int')
            stability_changes = np.concatenate([np.zeros((1,)), np.diff(stability)])
            idx_changes = np.sort(np.argwhere(stability_changes != 0))
            idx_changes = np.append(idx_changes, len(stability_changes))

            # create line segments
            lines, styles = [], []
            idx_old = 1
            for idx in idx_changes:
                lines.append(y[idx_old - 1:idx, :])
                styles.append(line_style_stable if stability[idx_old] else line_style_unstable)
                idx_old = idx

        else:

            lines = [y]
            styles = [line_style_stable]

        # create line collection
        array = kwargs.pop('array', 'x')
        line_col = Line3DCollection(segments=lines, linestyles=styles, **kwargs)

        # post-processing
        if array == 'x':
            array = x.squeeze()
        elif array == 'y':
            array = y[:, 1].squeeze()
        elif array == 'z':
            array = z.squeeze()
        line_col.set_array(array)

        return line_col

    @staticmethod
    def _get_axis_lims(x: np.array, padding: float = 0.) -> tuple:
        x_min, x_max = x.min(), x.max()
        x_pad = (x_max - x_min) * padding
        return x_min - x_pad, x_max + x_pad


def continue_period_doubling_bf(solution: dict, continuation: Union[str, int, Any], pyauto_instance: PyAuto,
                                max_iter: int = 1000, iteration: int = 0, **kwargs) -> tuple:
    """Automatically continue a cascade of period doubling bifurcations. Returns the labels of the continuation and the
    pyauto instance they were run on.

    Parameters
    ----------
    solution
    continuation
    pyauto_instance
    max_iter
    iteration
    kwargs

    Returns
    -------
    tuple
    """
    solutions = []
    i = 1
    for point, point_info in solution.items():
        if 'PD' in point_info['bifurcation']:
            s_tmp, cont = pyauto_instance.run(starting_point=f'PD{i}', name=f'pd_{iteration}', origin=continuation,
                                              **kwargs)
            solutions.append(f'pd_{iteration}')
            iteration += 1
            i += 1
            if iteration >= max_iter:
                break
            elif s_tmp:
                s_tmp2 = continue_period_doubling_bf(solution=s_tmp, continuation=cont, pyauto_instance=pyauto_instance,
                                                     iteration=iteration, **kwargs)
                solutions += s_tmp2
                iteration += len(s_tmp2)

    return solutions, pyauto_instance


def fractal_dimension(lyapunov_exponents: list) -> float:
    """Calculates the fractal or information dimension of an attractor of a dynamical system from its lyapunov
    epxonents, according to the Kaplan-Yorke formula (Kaplan and Yorke, 1979).

    Parameters
    ----------
    lyapunov_exponents
        List containing the lyapunov spectrum of a solution of a dynamical system.

    Returns
    -------
    float
        Fractal dimension of the attractor of the system.

    """

    LEs = np.sort(lyapunov_exponents)[::-1]
    if np.sum(LEs) > 0:
        return 0.0
    k = 0
    for j in range(len(LEs)-1):
        k = j+1
        if np.sum(LEs[:k]) < 0:
            k -= 1
            break
    return k + np.sum(LEs[:k]) / np.abs(LEs[k]) if k > 0 else 0.0


def get_point_idx(diag: list, point: int) -> int:
    """Extract list idx of correct diagnostics string for continuation point with index `point`.

    Parameters
    ----------
    diag
    point

    Returns
    -------
    int
        Point index for `diag`.

    """

    idx = point
    while idx < len(diag)-1:

        diag_tmp = diag[idx]['Text']
        if "Location of special point" in diag_tmp and "Convergence" not in diag_tmp:
            idx += 1
        else:

            # find index of line after first appearance of BR
            diag_tmp = diag_tmp.split('\n')
            idx_line = 1
            while idx_line < len(diag_tmp)-1:
                if 'BR' in diag_tmp[idx_line].split(' '):
                    break
                idx_line += 1
            diag_tmp = diag_tmp[idx_line+1]

            # find point number in text
            line_split = [d for d in diag_tmp.split(' ') if d != ""]
            if abs(int(line_split[1])) < point+1:
                idx += 1
            elif abs(int(line_split[1])) == point+1:
                return idx
            else:
                raise ValueError(f"Point with index {point+1} was not found on solution. Last auto output line that "
                                 f"was checked: \n {diag_tmp}")
    return idx
