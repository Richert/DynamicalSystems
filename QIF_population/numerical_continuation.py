import numpy as np
import matplotlib.pyplot as plt
from QIF import QIFMacro

# general parameters
T = 20.0
dt = 1e-4

# first parameter continuation
##############################

"""This serves the purpose of setting the adaptation strength to a desired value.
"""

a = np.linspace(0.0, 0.3, int(T/dt))
m = QIFMacro(J=15.0, eta_mean=-10.0, eta_fwhm=2.0, tau=1.0, tau_a=1.0)
v, r = m.run(T, dt, alpha=a)

plt.plot(v)
plt.figure()
plt.plot(r)
plt.show()

# second parameter continuation
###############################

"""Now, we look for a bifurcation when changing the excitability of the system via eta.
"""

POI = [-10.0, -5.90, -7.22, -6.28, -6.27]

for i, e in enumerate(POI[:-1]):
    eta = np.linspace(e, POI[i+1], int(T/dt))
    v, r = m.run(T, dt, eta=eta)
    plt.plot(v)
    plt.figure()
    plt.plot(r)
    plt.show()
