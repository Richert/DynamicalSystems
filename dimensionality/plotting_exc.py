import pickle
import matplotlib.pyplot as plt
plt.switch_backend('tkagg')
from seaborn import scatterplot, lineplot
import os
import sys
import numpy as np
from pandas import DataFrame

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
iv = "p"
condition = "dim2_exc"
path = "/home/richard-gast/Documents/data/dimensionality"

# load data
results = {"rep": [], "g": [], "Delta": [], iv: [], "dim_ss": [], "s_mean": [], "s_std": [], "s_norm": [],
           "dim_ir": [], "tau_ir": [], "offset_ir": [], "amp_ir": [], "tau_mf": []}
for file in os.listdir(path):
    if file[:len(condition)] == condition:

        # load data
        data = pickle.load(open(f"{path}/{file}", "rb"))

        # condition information
        f = file.split("_")
        if "N" in data:
            rep = int(f[-1].split(".")[0])
            results["rep"].append(rep)
            results["g"].append(data["g"])
            results["Delta"].append(data["Delta"])
            # results[iv].append(data[iv])
            results[iv].append(float(f[-2][1:]))

            # steady-state analysis
            results["dim_ss"].append(data["dim_ss"]*data["N"])
            results["s_mean"].append(np.mean(data["s_mean"])*1e3)
            results["s_std"].append(np.mean(data["s_std"]))
            results["s_norm"].append(results["s_std"][-1]*1e3/results["s_mean"][-1])

            # impulse response analysis
            results["dim_ir"].append(data["dim_ir"])
            results["tau_ir"].append(data["params_ir"][-2])
            results["offset_ir"].append(data["params_ir"][0])
            results["amp_ir"].append(data["params_ir"][2])
            results["tau_mf"].append(data["mf_params_ir"][-1])

# create dataframe
df = DataFrame.from_dict(results)

# plot of firing rate statistics
ivs = np.unique(df.loc[:, iv].values)
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(nrows=2, ncols=3)
for i, (y, ylabel) in enumerate(zip(["s_mean", "s_norm"], [r"$\bar s / \tau_s$", "$D(s)$"])):
    for j, p in enumerate(ivs):
        df_tmp = df.loc[df[iv] == p, :]
        ax = fig.add_subplot(grid[i, j])
        l = lineplot(df_tmp, x="g", hue="Delta", y=y, ax=ax, palette=cmap, legend=True if j == 2 else False)
        if i == 0:
            ax.set_title(rf"$p = {np.round(p * 1e-2, decimals=2)}$")
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
fig.suptitle("Steady-Sate Firing Rates of Excitatory Networks")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/exc_firing_rates.pdf')

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
            ax.set_title(fr"$p = {np.round(p*1e-2, decimals=2)}$")
            ax.set_xlabel("")
        else:
            ax.set_xlabel(r"conductance $g$ (nS)")
        if j == 2:
            leg = l.axes.get_legend()
            leg.set_title(r"$\Delta$ (mV)")
fig.suptitle("Dimensionality of Steady-State vs. Impulse Response")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/exc_dimensionality.pdf')

# scatter plots for network dimensionality vs. time constant
fig = plt.figure(figsize=(6, 5))
grid = fig.add_gridspec(nrows=2, ncols=2)
df_tmp = df.loc[df[iv] == ivs[0], :]
ax = fig.add_subplot(grid[1, 0])
lineplot(df_tmp, x="g", hue="Delta", y="dim_ss", ax=ax, palette=cmap, legend=False)
ax.set_ylabel(r"$D_{ss}$")
ax.set_xlabel(r"conductance $g$ (nS)")
ax.set_title("(A) (B) (C) (D) (E) (F)")
ax = fig.add_subplot(grid[0, 1])
l = lineplot(df_tmp, x="g", hue="Delta", y="dim_ir", ax=ax, palette=cmap, legend=True)
ax.set_ylabel(r"$D_{ir}$")
ax.set_xlabel(r"conductance $g$ (nS)")
ax.set_title("Impulse response dimensionality")
leg = l.axes.get_legend()
leg.set_title(r"$\Delta$ (mV)")
ax = fig.add_subplot(grid[1, 1])
lineplot(df_tmp, x="g", hue="Delta", y="tau_ir", ax=ax, palette=cmap, legend=False)
ax.set_ylabel(r"$\tau_{ir}$")
ax.set_xlabel(r"conductance $g$ (nS)")
ax.set_title("Impulse response time")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/exc_dimensionality_2.pdf')

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
            ax.set_ylabel(r"$D(C)$")
        else:
            ax.set_ylabel("")
        if i == 0:
            ax.set_title(fr"$p = {np.round(p*1e-2, decimals=2)}$")
        if i == 1:
            ax.set_xlabel(r"$D(s)$")
        else:
            ax.set_xlabel("")
        if j == 2:
            leg = s.axes.get_legend()
            leg.set_title(hue_title)
fig.suptitle("Steady-State Dimensionality")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/exc_dimensionality_ss.pdf')

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
            ax.set_title(fr"$p = {np.round(p*1e-2, decimals=2)}$")
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
plt.savefig(f'{path}/figures/exc_dimensionality_ir.pdf')

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
            ax.set_title(fr"$p = {np.round(p*1e-2, decimals=2)}$")
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
plt.savefig(f'{path}/figures/exc_dimensionality_tau.pdf')

plt.show()
