import numpy as np
import matplotlib.pyplot as plt

# parameters
inp = np.linspace(1.0, 100.0, num=1000)
v_r = -75.0
v_t = -40.0

# calculate corrected input
inp_corrected = inp*np.pi**2/(np.arctan(v_t/np.sqrt(inp)) - np.arctan(v_r/np.sqrt(inp)))**2

# calculate firing rates
f_corrected = np.sqrt(inp_corrected)/np.pi
f = np.sqrt(inp)/(np.arctan(v_t/np.sqrt(inp)) - np.arctan(v_r/np.sqrt(inp)))

# plotting
fig, ax = plt.subplots()
ax.plot(inp, f)
ax.plot(inp, f_corrected)
plt.legend(['MF_corrected', 'FRE'])
plt.show()
