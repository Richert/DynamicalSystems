import matplotlib.pyplot as plt
plt.switch_backend('tkagg')
from seaborn import scatterplot, lineplot
import numpy as np
from pandas import read_pickle

# figure settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 15.0
cmap = "ch:"

# condition
iv = "Delta_i"
iv_str = "\Delta_i"
iv_unit = "(mV)"
task_condition= "dim"
neuron_type = "eic"
condition = f"{task_condition}_{neuron_type}"
path = "/home/richard-gast/Documents/data/dimensionality"

# create dataframe
df = read_pickle(f"{path}/{condition}_summary.pkl")

# filter out parts of the parameter regime
df = df.loc[df["g"] > 0.0, :]
# df.loc[:, "dim_ss"] *= 1000
# df.loc[:, "dim_ir"] *= 1000

# plot of firing rate statistics
ivs = np.unique(df.loc[:, iv].values)
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(nrows=2, ncols=3)
for i, (y, ylabel) in enumerate(zip(["s_mean", "s_norm"], [r"$\bar s / \tau_s$", r"$\mathrm{std}(s) / \bar s$"])):
    for j, p in enumerate(ivs):
        df_tmp = df.loc[df[iv] == p, :]
        ax = fig.add_subplot(grid[i, j])
        l = lineplot(df_tmp, x="g", hue="Delta", y=y, ax=ax, palette=cmap, legend=True if j == 2 else False)
        if i == 0:
            ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
            ax.set_xlabel("")
        else:
            ax.set_xlabel(r"conductance $g$ (nS)")
        if j == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
        if j == 2:
            leg = l.axes.get_legend()
            leg.set_title(r"$\Delta$ (mV)")
fig.suptitle("Steady-Sate Firing Rates of E-I Networks")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/{neuron_type}_firing_rates.pdf')

# line plots for network dimensionality
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(nrows=2, ncols=len(ivs))
for j, p in enumerate(ivs):
    df_tmp = df.loc[df[iv] == p, :]
    for i, (y, ylabel) in enumerate(zip(["dim_ss", "dim_ir"], [r"$D_{ss}(C)$", r"$D_{ir}(C)$"])):
        ax = fig.add_subplot(grid[i, j])
        l = lineplot(df_tmp, x="g", hue="Delta", y=y, ax=ax, palette=cmap, legend=True if j == 2 else False)
        if j == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
        if i == 0:
            ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
            ax.set_xlabel("")
        else:
            ax.set_xlabel(r"conductance $g$ (nS)")
        if j == 2:
            leg = l.axes.get_legend()
            leg.set_title(r"$\Delta$ (mV)")
fig.suptitle("Dimensionality of Steady-State and Impulse Response")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/{neuron_type}_dimensionality.pdf')

# scatter plots for firing rate heterogeneity vs dimensionality in steady state
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(ncols=len(ivs), nrows=2)
for j, p in enumerate(ivs):
    df_tmp = df.loc[df[iv] == p, :]
    for i, (hue, hue_title) in enumerate(zip(["Delta", "g"], [r"$\Delta$ (mV)", r"$g$ (nS)"])):
        ax = fig.add_subplot(grid[i, j])
        s = scatterplot(df_tmp, x="s_norm", y="dim_ss", hue=hue, palette=cmap, legend=True if j == 2 else False,
                        ax=ax, s=markersize)
        if j == 0:
            ax.set_ylabel(r"$D_{ss}(C)$")
        else:
            ax.set_ylabel("")
        if i == 0:
            ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
        if i == 1:
            ax.set_xlabel(r"$\mathrm{std}(s) / \bar s$")
        else:
            ax.set_xlabel("")
        if j == 2:
            leg = s.axes.get_legend()
            leg.set_title(hue_title)
fig.suptitle("Steady-State Dimensionality")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/{neuron_type}_dimensionality_ss.pdf')

# filter results for scatter plots
min_tau = 20.0
df = df.loc[df["tau_ir"] >= min_tau, :]

# scatter plots for firing rate heterogeneity vs dimensionality
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(ncols=len(ivs), nrows=2)
for j, p in enumerate(ivs):
    df_tmp = df.loc[df[iv] == p, :]
    for i, (hue, hue_title) in enumerate(zip(["Delta", "g"], [r"$\Delta$ (mV)", r"$g$ (nS)"])):
        ax = fig.add_subplot(grid[i, j])
        s = scatterplot(df_tmp, x="dim_ir", y="dim_ss", hue=hue, palette=cmap, legend=True if j == 2 else False,
                        ax=ax, s=markersize)
        if j == 0:
            ax.set_ylabel(r"$D_{ss}(C)$")
        else:
            ax.set_ylabel("")
        if i == 0:
            ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
        if i == 1:
            ax.set_xlabel(r"$D_{ir}(C)$")
        else:
            ax.set_xlabel("")
        if j == 2:
            leg = s.axes.get_legend()
            leg.set_title(hue_title)
fig.suptitle("Steady-State vs. Impulse Response Dimensionality")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/{neuron_type}_dimensionality_ir.pdf')

# scatter plots for impulse response time constant vs dimensionality
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(ncols=len(ivs), nrows=2)
for j, p in enumerate(ivs):
    df_tmp = df.loc[df[iv] == p, :]
    for i, (hue, hue_title) in enumerate(zip(["Delta", "g"], [r"$\Delta$ (mV)", r"$g$ (nS)"])):
        ax = fig.add_subplot(grid[i, j])
        s = scatterplot(df_tmp, x="dim_ir", y="tau_ir", hue=hue, palette=cmap, legend=True if j == 2 else False,
                        ax=ax, s=markersize)
        if j == 0:
            ax.set_ylabel(r"$\tau_{ir}$ (ms)")
        else:
            ax.set_ylabel("")
        if i == 0:
            ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
        if i == 1:
            ax.set_xlabel(r"$D_{ir}(C)$")
        else:
            ax.set_xlabel("")
        if j == 2:
            leg = s.axes.get_legend()
            leg.set_title(hue_title)
fig.suptitle("Impulse Response Dimensionality vs. Decay Time Constant")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/{neuron_type}_dimensionality_tau.pdf')

# scatter plots for centering effects on dimensionality
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(ncols=len(ivs), nrows=2)
for j, p in enumerate(ivs):
    df_tmp = df.loc[df[iv] == p, :]
    for i, (hue, hue_title) in enumerate(zip(["Delta", "g"], [r"$\Delta$ (mV)", r"$g$ (nS)"])):
        ax = fig.add_subplot(grid[i, j])
        s = scatterplot(df_tmp, x="dim_ss", y="dim_ss_nc", hue=hue, palette=cmap, legend=True if j == 2 else False,
                        ax=ax, s=markersize)
        if j == 0:
            ax.set_ylabel(r"$D_{ss}(C_{nc})$")
        else:
            ax.set_ylabel("")
        if i == 0:
            ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
        if i == 1:
            ax.set_xlabel(r"$D_{ss}(C)$")
        else:
            ax.set_xlabel("")
        if j == 2:
            leg = s.axes.get_legend()
            leg.set_title(hue_title)
fig.suptitle("Centered vs. non-centered Steady-State Dimensionality")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/{neuron_type}_centering_ss.pdf')

fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(ncols=len(ivs), nrows=2)
for j, p in enumerate(ivs):
    df_tmp = df.loc[df[iv] == p, :]
    for i, (hue, hue_title) in enumerate(zip(["Delta", "g"], [r"$\Delta$ (mV)", r"$g$ (nS)"])):
        ax = fig.add_subplot(grid[i, j])
        s = scatterplot(df_tmp, x="dim_ir", y="dim_ir_nc", hue=hue, palette=cmap, legend=True if j == 2 else False,
                        ax=ax, s=markersize)
        if j == 0:
            ax.set_ylabel(r"$D_{ir}(C_{nc})$")
        else:
            ax.set_ylabel("")
        if i == 0:
            ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
        if i == 1:
            ax.set_xlabel(r"$D_{ir}(C)$")
        else:
            ax.set_xlabel("")
        if j == 2:
            leg = s.axes.get_legend()
            leg.set_title(hue_title)
fig.suptitle("Centered vs. non-centered Impulse Response Dimensionality")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/{neuron_type}_centering_ir.pdf')

plt.show()
