import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import numpy as np

# preparations
##############

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0

# load data
data_control = pickle.load(open("dSPN_control.pkl", "rb"))
data_gabazine = pickle.load(open("dSPN_gabazine.pkl", "rb"))

# postprocessing
################

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=3, ncols=3, figure=fig)

# plot histograms
hist_params = ["d", "tau_s", "tau_f"]
titles = ["Delay to EPSP onset", "Slow decay time constant", "Fast decay time constant"]
for idx, (p, t) in enumerate(zip(hist_params, titles)):
    ax = fig.add_subplot(grid[0, idx])
    ax.hist(data_control["parameters"].loc[p, :].values, bins=10, density=False, histtype="bar", color="blue",
            label="control")
    ax.hist(data_gabazine["parameters"].loc[p, :].values, bins=10, density=False, histtype="bar", color="orange",
            label="gabazine", alpha=0.5)
    ax.set_xlabel(rf"${p}$")
    ax.set_ylabel("count")
    ax.set_title(t)
    plt.legend()

# plot exemplary fits of control condition
control_trials = np.random.randint(0, 9, size=3)
print_params = ["tau_s", "tau_f"]
for idx, trial in enumerate(control_trials):
    ax = fig.add_subplot(grid[1, idx])
    ax.plot(data_control["target_epsps"].iloc[:, trial].values, color="black", label="target")
    ax.plot(data_control["fitted_epsps"].iloc[:, trial].values, color="blue", label="fit")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("EPSP (mV)")
    param_vals = [data_control["parameters"].loc[p, trial] for p in print_params]
    param_str = ', '.join([fr'${p} = {int(np.round(v, decimals=0))}$' for p, v in zip(print_params, param_vals)])
    ax.set_title(f"Control: {param_str}")
    plt.legend()

# plot exemplary fits of gabazine condition
control_trials = np.random.randint(0, 9, size=3)
print_params = ["tau_s", "tau_f"]
for idx, trial in enumerate(control_trials):
    ax = fig.add_subplot(grid[2, idx])
    ax.plot(data_control["target_epsps"].iloc[:, trial].values, color="black", label="target")
    ax.plot(data_control["fitted_epsps"].iloc[:, trial].values, color="orange", label="fit")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("EPSP (mV)")
    param_vals = [data_control["parameters"].loc[p, trial] for p in print_params]
    param_str = ', '.join([fr'${p} = {int(np.round(v, decimals=0))}$' for p, v in zip(print_params, param_vals)])
    ax.set_title(f"Gabazine: {param_str}")
    plt.legend()

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'epsp_fitting.pdf')
plt.show()
