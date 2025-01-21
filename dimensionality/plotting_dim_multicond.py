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
savefig = False

# condition
iv = "spatial_dim"
iv_str = "model"
iv_unit = "D"
task_condition= "eic"
neuron_type = "eic"
condition = f"{task_condition}_{neuron_type}"
path = "/home/richard-gast/Documents/data/dimensionality"

# create dataframe
df = read_pickle(f"{path}/{task_condition}_summary.pkl")

# filter out parts of the parameter regime
# df = df.loc[df["g"] > 0.0, :]
# df.loc[:, "dim_ss"] *= 1000
# df.loc[:, "dim_ir"] *= 1000

# plot of firing rate statistics
fr_vars = ["s_mean", "s_norm", "ff_mean"]
fr_labels = [r"$\bar s / \tau_s$ (Hz)", r"$\mathrm{std}(s) / \bar s$ (Hz)", r"$\mathrm{ff}$"]
ivs = np.unique(df.loc[:, iv].values)
fig = plt.figure(figsize=(12, 2*len(fr_vars)))
grid = fig.add_gridspec(nrows=len(fr_vars), ncols=3)
for i, (y, ylabel) in enumerate(zip(fr_vars, fr_labels)):
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
if savefig:
    plt.savefig(f'{path}/figures/{neuron_type}_firing_rates.pdf')

# line plots for steady-state network dimensionality
dim_types = ["dim_ss", "dim_ss_c", "dim_ss_rc"]
dim_labels = [r"$D_{ss}(C)$", r"$D_{ss}(C_r)$", r"$D_{ss}(C_{rc})$"]
fig = plt.figure(figsize=(12, 2*len(dim_types)))
grid = fig.add_gridspec(nrows=len(dim_types), ncols=len(ivs))
for j, p in enumerate(ivs):
    df_tmp = df.loc[df[iv] == p, :]
    for i, (y, ylabel) in enumerate(zip(dim_types, dim_labels)):
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
fig.suptitle("Dimensionality of Steady-State")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/{neuron_type}_dimensionality_ss.pdf')

# line plots for impulse response dimensionality
dim_types = ["dim_sep", "dim_sep_c", "dim_sep_rc"]
dim_labels = [r"$D_{ir}(C)$", r"$D_{ir}(C_r)$", r"$D_{ir}(C_{rc})$"]
fig = plt.figure(figsize=(12, 2*len(dim_types)))
grid = fig.add_gridspec(nrows=len(dim_types), ncols=len(ivs))
for j, p in enumerate(ivs):
    df_tmp = df.loc[df[iv] == p, :]
    for i, (y, ylabel) in enumerate(zip(dim_types, dim_labels)):
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
fig.suptitle("Dimensionality of Impulse Response")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/{neuron_type}_dimensionality_ir.pdf')

# line plots for impulse response statistics
dim_types = ["tau_ir", "offset_ir", "amp_ir"]
dim_labels = [r"$\tau_{ir}$", r"$c_{ir}$", r"$a_{ir}$"]
fig = plt.figure(figsize=(12, 2*len(dim_types)))
grid = fig.add_gridspec(nrows=len(dim_types), ncols=len(ivs))
for j, p in enumerate(ivs):
    df_tmp = df.loc[df[iv] == p, :]
    for i, (y, ylabel) in enumerate(zip(dim_types, dim_labels)):
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
fig.suptitle("Impulse Response Properties")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/{neuron_type}_impulse_response.pdf')

# line plots for kernel statistics
dim_types = ["tau_k", "mean_k", "std_k"]
dim_labels = [r"$\tau_{k}$", r"$\mu_{k}$", r"$\sigma_{k}$"]
fig = plt.figure(figsize=(12, 2*len(dim_types)))
grid = fig.add_gridspec(nrows=len(dim_types), ncols=len(ivs))
for j, p in enumerate(ivs):
    df_tmp = df.loc[df[iv] == p, :]
    for i, (y, ylabel) in enumerate(zip(dim_types, dim_labels)):
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
fig.suptitle("Kernel Properties")
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
fig.canvas.draw()
plt.savefig(f'{path}/figures/{neuron_type}_kernel.pdf')

# # scatter plots for firing rate heterogeneity vs dimensionality in steady state
# dim_types = ["dim_ss", "dim_sep", "dim_ss_rc", "dim_sep_rc"]
# dim_labels = [r"$D_{ss}(C)$", r"$D_{ir}(C)$", r"$D_{ss}(C_r)$", r"$D_{ir}(C_r)$"]
# for dim_type, dim_label in zip(dim_types, dim_labels):
#     fig = plt.figure(figsize=(12, 6))
#     grid = fig.add_gridspec(ncols=len(ivs), nrows=2)
#     for j, p in enumerate(ivs):
#         df_tmp = df.loc[df[iv] == p, :]
#         for i, (hue, hue_title) in enumerate(zip(["Delta", "g"], [r"$\Delta$ (mV)", r"$g$ (nS)"])):
#             ax = fig.add_subplot(grid[i, j])
#             s = scatterplot(df_tmp, x="s_norm", y=dim_type, hue=hue, palette=cmap, legend=True if j == 2 else False,
#                             ax=ax, s=markersize)
#             if j == 0:
#                 ax.set_ylabel(dim_label)
#             else:
#                 ax.set_ylabel("")
#             if i == 0:
#                 ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
#             if i == 1:
#                 ax.set_xlabel(r"$\mathrm{std}(s) / \bar s$")
#             else:
#                 ax.set_xlabel("")
#             if j == 2:
#                 leg = s.axes.get_legend()
#                 leg.set_title(hue_title)
#     fig.suptitle(f"Firing Rate Variability vs. Dimensionality: {dim_type}")
#     fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
#     fig.canvas.draw()
#     plt.savefig(f'{path}/figures/{neuron_type}_ratevar_{dim_type}.pdf')
#
# # scatter plots for impulse response time constant vs dimensionality
# dim_types = ["dim_ss", "dim_sep", "dim_ss_rc", "dim_sep_rc"]
# dim_labels = [r"$D_{ss}(C)$", r"$D_{ir}(C)$", r"$D_{ss}(C_r)$", r"$D_{ir}(C_r)$"]
# for dim_type, dim_label in zip(dim_types, dim_labels):
#     fig = plt.figure(figsize=(12, 6))
#     grid = fig.add_gridspec(ncols=len(ivs), nrows=2)
#     for j, p in enumerate(ivs):
#         df_tmp = df.loc[df[iv] == p, :]
#         for i, (hue, hue_title) in enumerate(zip(["Delta", "g"], [r"$\Delta$ (mV)", r"$g$ (nS)"])):
#             ax = fig.add_subplot(grid[i, j])
#             s = scatterplot(df_tmp, x="tau_ir", y=dim_type, hue=hue, palette=cmap, legend=True if j == 2 else False,
#                             ax=ax, s=markersize)
#             if j == 0:
#                 ax.set_ylabel(dim_label)
#             else:
#                 ax.set_ylabel("")
#             if i == 0:
#                 ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
#             if i == 1:
#                 ax.set_xlabel(r"$\tau_{ir}$ (ms)")
#             else:
#                 ax.set_xlabel("")
#             if j == 2:
#                 leg = s.axes.get_legend()
#                 leg.set_title(hue_title)
#     fig.suptitle(f"Impulse Response Decay Time vs. Dimensionality: {dim_type}")
#     fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
#     fig.canvas.draw()
#     plt.savefig(f'{path}/figures/{neuron_type}_tau_{dim_type}.pdf')
#
# # scatter plots for kernel quality vs dimensionality
# dim_types = ["dim_ss", "dim_sep", "dim_ss_rc", "dim_sep_rc"]
# dim_labels = [r"$D_{ss}(C)$", r"$D_{ir}(C)$", r"$D_{ss}(C_r)$", r"$D_{ir}(C_r)$"]
# for dim_type, dim_label in zip(dim_types, dim_labels):
#     fig = plt.figure(figsize=(12, 6))
#     grid = fig.add_gridspec(ncols=len(ivs), nrows=2)
#     for j, p in enumerate(ivs):
#         df_tmp = df.loc[df[iv] == p, :]
#         for i, (hue, hue_title) in enumerate(zip(["Delta", "g"], [r"$\Delta$ (mV)", r"$g$ (nS)"])):
#             ax = fig.add_subplot(grid[i, j])
#             s = scatterplot(df_tmp, x="mean_k", y=dim_type, hue=hue, palette=cmap, legend=True if j == 2 else False,
#                             ax=ax, s=markersize)
#             if j == 0:
#                 ax.set_ylabel(dim_label)
#             else:
#                 ax.set_ylabel("")
#             if i == 0:
#                 ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
#             if i == 1:
#                 ax.set_xlabel(r"$\mathrm{mean}(k)$")
#             else:
#                 ax.set_xlabel("")
#             if j == 2:
#                 leg = s.axes.get_legend()
#                 leg.set_title(hue_title)
#     fig.suptitle(f"Kernel Quality vs. Dimensionality: {dim_type}")
#     fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
#     fig.canvas.draw()
#     plt.savefig(f'{path}/figures/{neuron_type}_kernel_{dim_type}.pdf')

# # scatter plots for firing rate heterogeneity vs dimensionality
# fig = plt.figure(figsize=(12, 6))
# grid = fig.add_gridspec(ncols=len(ivs), nrows=2)
# for j, p in enumerate(ivs):
#     df_tmp = df.loc[df[iv] == p, :]
#     for i, (hue, hue_title) in enumerate(zip(["Delta", "g"], [r"$\Delta$ (mV)", r"$g$ (nS)"])):
#         ax = fig.add_subplot(grid[i, j])
#         s = scatterplot(df_tmp, x="dim_ir", y="dim_ss", hue=hue, palette=cmap, legend=True if j == 2 else False,
#                         ax=ax, s=markersize)
#         if j == 0:
#             ax.set_ylabel(r"$D_{ss}(C)$")
#         else:
#             ax.set_ylabel("")
#         if i == 0:
#             ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
#         if i == 1:
#             ax.set_xlabel(r"$D_{ir}(C)$")
#         else:
#             ax.set_xlabel("")
#         if j == 2:
#             leg = s.axes.get_legend()
#             leg.set_title(hue_title)
# fig.suptitle("Steady-State vs. Impulse Response Dimensionality")
# fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
# fig.canvas.draw()
# plt.savefig(f'{path}/figures/{neuron_type}_dimensionality_ir.pdf')
#
# # scatter plots for impulse response time constant vs dimensionality
# fig = plt.figure(figsize=(12, 6))
# grid = fig.add_gridspec(ncols=len(ivs), nrows=2)
# for j, p in enumerate(ivs):
#     df_tmp = df.loc[df[iv] == p, :]
#     for i, (hue, hue_title) in enumerate(zip(["Delta", "g"], [r"$\Delta$ (mV)", r"$g$ (nS)"])):
#         ax = fig.add_subplot(grid[i, j])
#         s = scatterplot(df_tmp, x="dim_ir", y="tau_ir", hue=hue, palette=cmap, legend=True if j == 2 else False,
#                         ax=ax, s=markersize)
#         if j == 0:
#             ax.set_ylabel(r"$\tau_{ir}$ (ms)")
#         else:
#             ax.set_ylabel("")
#         if i == 0:
#             ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
#         if i == 1:
#             ax.set_xlabel(r"$D_{ir}(C)$")
#         else:
#             ax.set_xlabel("")
#         if j == 2:
#             leg = s.axes.get_legend()
#             leg.set_title(hue_title)
# fig.suptitle("Impulse Response Dimensionality vs. Decay Time Constant")
# fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
# fig.canvas.draw()
# plt.savefig(f'{path}/figures/{neuron_type}_dimensionality_tau.pdf')
#
# # scatter plots for centering effects on dimensionality
# fig = plt.figure(figsize=(12, 6))
# grid = fig.add_gridspec(ncols=len(ivs), nrows=2)
# for j, p in enumerate(ivs):
#     df_tmp = df.loc[df[iv] == p, :]
#     for i, (hue, hue_title) in enumerate(zip(["Delta", "g"], [r"$\Delta$ (mV)", r"$g$ (nS)"])):
#         ax = fig.add_subplot(grid[i, j])
#         s = scatterplot(df_tmp, x="dim_ss", y="dim_ss_nc", hue=hue, palette=cmap, legend=True if j == 2 else False,
#                         ax=ax, s=markersize)
#         if j == 0:
#             ax.set_ylabel(r"$D_{ss}(C_{nc})$")
#         else:
#             ax.set_ylabel("")
#         if i == 0:
#             ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
#         if i == 1:
#             ax.set_xlabel(r"$D_{ss}(C)$")
#         else:
#             ax.set_xlabel("")
#         if j == 2:
#             leg = s.axes.get_legend()
#             leg.set_title(hue_title)
# fig.suptitle("Centered vs. non-centered Steady-State Dimensionality")
# fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
# fig.canvas.draw()
# plt.savefig(f'{path}/figures/{neuron_type}_centering_ss.pdf')
#
# fig = plt.figure(figsize=(12, 6))
# grid = fig.add_gridspec(ncols=len(ivs), nrows=2)
# for j, p in enumerate(ivs):
#     df_tmp = df.loc[df[iv] == p, :]
#     for i, (hue, hue_title) in enumerate(zip(["Delta", "g"], [r"$\Delta$ (mV)", r"$g$ (nS)"])):
#         ax = fig.add_subplot(grid[i, j])
#         s = scatterplot(df_tmp, x="dim_ir", y="dim_ir_nc", hue=hue, palette=cmap, legend=True if j == 2 else False,
#                         ax=ax, s=markersize)
#         if j == 0:
#             ax.set_ylabel(r"$D_{ir}(C_{nc})$")
#         else:
#             ax.set_ylabel("")
#         if i == 0:
#             ax.set_title(rf"${iv_str} = {np.round(p, decimals=2)}$ {iv_unit}")
#         if i == 1:
#             ax.set_xlabel(r"$D_{ir}(C)$")
#         else:
#             ax.set_xlabel("")
#         if j == 2:
#             leg = s.axes.get_legend()
#             leg.set_title(hue_title)
# fig.suptitle("Centered vs. non-centered Impulse Response Dimensionality")
# fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
# fig.canvas.draw()
# plt.savefig(f'{path}/figures/{neuron_type}_centering_ir.pdf')

plt.show()
