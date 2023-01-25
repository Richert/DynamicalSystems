"""
This file contains definitions for kernel functions that may be used to fit post-synaptic response profiles.
Import these functions into the python file where the fitting is supposed to take place via:

    `from kernels import <function_name>`

They can be used as functions to optimize with `scipy.optimize.curve_fit`, for example.
"""
import numpy as np
from typing import Union


def exponential(t: Union[float, np.ndarray], g: float, tau: float):
    """Mono-exponential kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param g: Scaling of the kernel function. Must be a scalar.
    :param tau: Decay time constant of the kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    return g * np.exp(-t / tau)


def alpha(t: Union[float, np.ndarray], g: float, tau: float):
    """Alpha kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param g: Scaling of the kernel function. Must be a scalar.
    :param tau: Time constant of the alpha kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    return g * t * np.exp(1 - t / tau) / tau


def biexponential(t: Union[float, np.ndarray], g: float, tau_r: float, tau_d: float):
    """Bi-exponential kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param g: Scaling of the kernel function. Must be a scalar.
    :param tau_r: Rise time constant of the kernel. Must be a scalar.
    :param tau_d: Decay time constant of the kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    return g * tau_d * tau_r * (np.exp(-t / tau_d) - np.exp(-t / tau_r)) / (tau_d - tau_r)


def dualexponential(t: Union[float, np.ndarray], d: float, g: float, a: float, tau_r: float, tau_s: float,
                    tau_f: float):
    """Bi-exponential kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param d: Delay until onset of the kernel response. Must be a scalar.
    :param g: Scaling of the kernel function. Must be a scalar.
    :param tau_r: Rise time constant of the kernel. Must be a scalar.
    :param tau_s: Slow decay time constant of the kernel. Must be a scalar.
    :param tau_f: Fast decay time constant of the kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    t1 = t - d
    on = 1.0 * (t1 > 0.0)
    return on * g * (1 - np.exp(-t1 / tau_r)) * (a * np.exp(-t1 / tau_s) + (1 - a) * np.exp(-t1 / tau_f))


def two_biexponential(t: Union[float, np.ndarray], g1: float, g2: float, tau_r1: float, tau_r2: float, tau_d1: float,
                      tau_d2: float):
    """Bi-exponential kernel function.

    :param t: time in arbitrary units. Can be a vector or a single scalar.
    :param g1: Scaling of the first kernel function. Must be a scalar.
    :param tau_r1: Rise time constant of the first kernel. Must be a scalar.
    :param tau_d1: Decay time constant of the first kernel. Must be a scalar.
    :param g2: Scaling of the second kernel function. Must be a scalar.
    :param tau_r2: Rise time constant of the second kernel. Must be a scalar.
    :param tau_d2: Decay time constant of the second kernel. Must be a scalar.
    :return: Value of the kernel function at each entry in `t`.
    """
    return biexponential(t, g1, tau_r1, tau_d1) + biexponential(t, g2, tau_r2, tau_d2)
