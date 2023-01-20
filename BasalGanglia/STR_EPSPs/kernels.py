"""
This file contains definitions for kernel functions that may be used to fit post-synaptic response profiles.
Import these functions into the python file where the fitting is supposed to take place via:

    `from kernels import <function_name>`

They can be used as functions to optimize with `scipy.optimize.curve_fit`, for example.
"""
import numpy as np
from typing import Union


def exponential(t: Union[float, np.ndarray], a: float, tau: float):
    """Mono-exponential kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param a: Scaling of the kernel function. Must be a scalar.
    :param tau: Decay time constant of the kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    return a*np.exp(-t/tau)


def alpha(t: Union[float, np.ndarray], a: float, tau: float):
    """Alpha kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param a: Scaling of the kernel function. Must be a scalar.
    :param tau: Time constant of the alpha kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    return a*t*np.exp(1-t/tau)/tau


def biexponential(t: Union[float, np.ndarray], a: float, tau_r: float, tau_d: float):
    """Bi-exponential kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param a: Scaling of the kernel function. Must be a scalar.
    :param tau_r: Rise time constant of the kernel. Must be a scalar.
    :param tau_d: Decay time constant of the kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    return a*(np.exp(-t/tau_r) - np.exp(-t/tau_d))


def dualexponential(t: Union[float, np.ndarray], a: float, b: float, tau_r: float, tau_slow: float, tau_fast: float):
    """Bi-exponential kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param a: Scaling of the kernel function. Must be a scalar.
    :param tau_r: Rise time constant of the kernel. Must be a scalar.
    :param tau_d: Decay time constant of the kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    return a * (1 - np.exp(-t/tau_r)) * (b*np.exp(-t/tau_fast) + (1-b)*np.exp(-t/tau_slow))
