from kernels import exponential, alpha, biexponential, dualexponential, alphas_combined, biexponential_combined
import matplotlib.pyplot as plt
import numpy as np

# parameters
n = 10000
tau_r, tau_d = 0.09, 0.005
a = 1.0

# generate artifical data
time = np.linspace(0, 0.5, n)
data = biexponential(time, a, tau_r, tau_d)

# plot artificial data
plt.plot(time, data)
plt.xlabel("time (s)")
plt.ylabel("EPSP (mV)")
plt.show()
