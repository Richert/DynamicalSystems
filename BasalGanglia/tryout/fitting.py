import numpy as np
from scipy.optimize import curve_fit
from kernels import exponential, alpha, biexponential, dualexponential
import matplotlib.pyplot as plt


# parameters
n = 10000
tau_r, tau_slow, tau_fast = 4.0, 20.0, 6.0
a, b = 2.0, 0.2

# generate artifical data
noise = 0.02
time = np.linspace(0, 500.0, n)
data = dualexponential(time, a, b, tau_r, tau_slow, tau_fast) + np.random.uniform(size=(n,))*noise

# plot artificial data
plt.plot(data)
plt.show()

# fit any model to the data
params, _ = curve_fit(alpha, time, data, p0=[1.0, 2.0], bounds=([0.0, 0.0], [10.0, 50.0]), full_output=False)

# plot fitted data against target data
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(alpha(time, params[0], params[1]), "blue", label="fitted")
ax.plot(data, "orange", label="target")
ax.set_xlabel("time")
ax.set_ylabel("signal")
plt.title(f"Fitted parameters: {params}")
plt.show()
