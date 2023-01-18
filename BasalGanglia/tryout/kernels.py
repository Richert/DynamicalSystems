import numpy as np
from typing import Union


def exponential(t: Union[float, np.ndarray], a: float, tau: float):
    return a*np.exp(-t/tau)


def alpha(t: Union[float, np.ndarray], a: float, tau: float):
    return a*t*np.exp(1-t/tau)/tau


def biexponential(t: Union[float, np.ndarray], a: float, tau_r: float, tau_d: float):
    return a*(np.exp(-t/tau_r) - np.exp(-t/tau_d))


def dualexponential(t: Union[float, np.ndarray], a: float, b: float, tau_r: float, tau_slow: float, tau_fast: float):
    return a * (1 - np.exp(-t/tau_r)) * (b*np.exp(-t/tau_fast) + (1-b)*np.exp(-t/tau_slow))
