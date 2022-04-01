import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (4.5, 1.5)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['axes.titlepad'] = 1.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 15
cmap = sns.color_palette("plasma", as_cmap=False, n_colors=4)

# create time series
N = 1000
f = 4
threshold = 0.1
x = np.linspace(0, 1, N)
y = np.sin(x*2*np.pi*f)
sigmoid = lambda x: 0 if x < threshold else 1/(1 + np.exp(-10*(x - 0.6)))
z = np.asarray([sigmoid(x_tmp) for x_tmp in x])

# plotting
fig, ax = plt.subplots()
ax.plot(x, y*z)
ax.set_xlabel(r"$A, B, C, D$")
ax.set_ylabel(r"$\theta$, $\sigma$")
ax.set_title(r"$\theta_i$")

# saving
fig.canvas.draw()
plt.savefig(f'phase_oscillator.svg')
plt.show()
