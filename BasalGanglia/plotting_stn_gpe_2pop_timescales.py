from matplotlib import gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyrates.utility.pyauto import PyAuto
from pandas import DataFrame
import os
import h5py
import pickle
import sys
sys.path.append('../')

################
# preparations #
################

# plot settings
###############

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (5.25, 5.0)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['axes.titlepad'] = 0.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 25
cbar_shrink = 0.7
vmin = 10
vmax = 100
cm = plt.get_cmap('RdBu')

# data for subfigure A
#######################

fname = 'stn_gpe_2pop'

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/{fname}_timescales_conts.pkl", auto_dir=auto_dir)

# data for subfigure B
######################

directories = ["/home/rgast/JuliaProjects/JuRates/BasalGanglia/results/stn_gpe_beta_results"]
fid = "stn_gpe_beta"
dv = 'p'
ivs = ["tau_e", "tau_p", "tau_ampa_r", "tau_ampa_d", "tau_gabaa_r", "tau_gabaa_d", "tau_stn"]

# load fitting data into frame
df = DataFrame(data=[[0.0, 0.0, '', '', 0]], columns=['value', 'fitness', 'parameter', 'model', 'index'])
for d in directories:
    for fn in os.listdir(d):
        if fn.startswith(fid) and fn.endswith("params.h5"):
            f = h5py.File(f"{d}/{fn}", 'r')
            fn_split = fn.split('_')
            f2 = h5py.File(f"{d}/{'_'.join(fn_split[:-1] + ['fitness.h5'])}", 'r')
            index = int(fn_split[-2])
            fitness = f2["f"][()]
            for key in ivs:
                df_tmp = DataFrame(data=[[f[dv][key][()], fitness, key, 'beta', index]],
                                   columns=['value', 'fitness', 'parameter', 'model', 'index'])
                df = df.append(df_tmp)
df = df.iloc[1:, :]

# add dataframe with original model data
vals = [13.0, 25.0, 0.8, 3.7, 0.5, 5.0, 2.0]
data = [[v, 0.0, p, 'gamma', i] for i, (v, p) in enumerate(zip(vals, ivs))]
df2 = DataFrame(data=data, columns=['value', 'fitness', 'parameter', 'model', 'index'])
df = df.append(df2)

# load pyauto data
a2 = PyAuto.from_file(f"results/{fname}_beta_conts.pkl", auto_dir=auto_dir)

# load simulation data
beta_data = pickle.load(open(f"results/{fname}_beta_sims.p", "rb"))

############
# plotting #
############

data = a.additional_attributes

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=5, ncols=9, figure=fig)

# subplot A
###########

# 2d: tau_e x tau_p
ax1 = fig.add_subplot(grid[0:2, 0:3])
df1 = pd.DataFrame(np.ma.masked_where(data['tau_e_p_periods'] == 0, data['tau_e_p_periods']),
                   columns=data['tau_e'], index=data['tau_p'])
xmin1, xmax1 = np.min(data['tau_e']), np.max(data['tau_e'])
ymin1, ymax1 = np.min(data['tau_p']), np.max(data['tau_p'])
im1 = ax1.imshow(df1,
                 extent=(xmin1, xmax1, ymin1, ymax1),
                 origin='lower',
                 aspect=(xmax1-xmin1)/(ymax1-ymin1),
                 vmin=vmin, vmax=vmax, cmap=cm
                 )
ax1 = a.plot_continuation('PAR(19)', 'PAR(20)', cont='tau_e/tau_p:hb1', line_style_unstable='solid', ignore=['UZ'],
                          ax=ax1, line_color_stable='#ee2b2b')
ax1.set_xlim([xmin1, xmax1])
ax1.set_ylim([ymin1, ymax1])
ax1.set_xlabel(r'$\tau_e$')
ax1.set_ylabel(r'$\tau_p$')
ax1.set_title('A', loc='left')
# fig.colorbar(im1, ax=ax1, shrink=cbar_shrink)

# 2d: tau_ampa x tau_gabaa
ax2 = fig.add_subplot(grid[0:2, 3:6])
df2 = pd.DataFrame(np.ma.masked_where(data['tau_ampa_gabaa_periods'] == 0, data['tau_ampa_gabaa_periods']),
                   columns=data['tau_ampa'], index=data['tau_gabaa'])
xmin2, xmax2 = np.min(data['tau_ampa']), np.max(data['tau_ampa'])
ymin2, ymax2 = np.min(data['tau_gabaa']), np.max(data['tau_gabaa'])
im2 = ax2.imshow(df2,
                 extent=(xmin2, xmax2, ymin2, ymax2),
                 origin='lower',
                 aspect=(xmax2 - xmin2) / (ymax2 - ymin2),
                 vmin=vmin, vmax=vmax, cmap=cm
                 )
ax2 = a.plot_continuation('PAR(21)', 'PAR(22)', cont='tau_ampa_d/tau_gabaa_d:hb1', line_style_unstable='solid',
                          ignore=['UZ'], ax=ax2, line_color_stable='#ee2b2b')
ax2.set_xlim([3.7, xmax2])
ax2.set_ylim([ymin2, ymax2])
ax2.set_xlabel(r'$\tau_{\mathrm{ampa}}$')
ax2.set_ylabel(r'$\tau_{\mathrm{gabaa}}$')
# fig.colorbar(im2, ax=ax2, shrink=cbar_shrink)

# 2d: tau_ampa x tau_gabaa
ax3 = fig.add_subplot(grid[0:2, 6:])
df3 = pd.DataFrame(np.ma.masked_where(data['tau_stn_gabaa_periods'] == 0, data['tau_stn_gabaa_periods']),
                   columns=data['tau_stn'], index=data['tau_gabaa2'])
xmin3, xmax3 = np.min(data['tau_stn']), np.max(data['tau_stn'])
ymin3, ymax3 = np.min(data['tau_gabaa2']), np.max(data['tau_gabaa2'])
im3 = ax3.imshow(df3,
                 extent=(xmin3, xmax3, ymin3, ymax3),
                 origin='lower',
                 aspect=(xmax3 - xmin3)/(ymax3 - ymin3),
                 vmin=vmin, vmax=vmax, cmap=cm
                 )
ax3 = a.plot_continuation('PAR(18)', 'PAR(22)', cont='tau_gabaa_d/tau_stn:hb1', line_style_unstable='solid',
                          ignore=['UZ'], ax=ax3, line_color_stable='#ee2b2b')
ax3.set_xlim([xmin3, xmax3])
ax3.set_ylim([ymin3, ymax3])
ax3.set_xlabel(r'$\frac{\tau_{\mathrm{gabaa}}^e}{\tau_{\mathrm{gabaa}}^p}$')
ax3.set_ylabel(r'$\tau_{\mathrm{gabaa}}$')
fig.colorbar(im3, ax=ax3, shrink=cbar_shrink)

# subplot B
###########

# barplot
ax4 = fig.add_subplot(grid[2:4, 0:5])
ax4 = sns.barplot(data=df.loc[df['fitness'] < 2, :], x='parameter', y='value', hue='model', ax=ax4)
ax4.set_xticklabels([r'$\tau_e$', r'$\tau_p$', r'$\tau_{\mathrm{ampa}}^r$', r'$\tau_{\mathrm{ampa}}^d$',
                     r'$\tau_{\mathrm{gabaa}}^r$', r'$\tau_{\mathrm{gabaa}}^d$',
                     r'$\frac{\tau_{\mathrm{gabaa}}^e}{\tau_{\mathrm{gabaa}}^p}$'])
ax4.set_xlabel('')
ax4.set_title('B')

# 2D bifurcation diagram
ax5 = fig.add_subplot(grid[2:4, 5:])
ax5 = a2.plot_continuation('PAR(15)', 'PAR(4)', cont='k_gp/k_pe:hb1', line_style_unstable='solid', ignore=['UZ'],
                           ax=ax5)
ax5 = a2.plot_continuation('PAR(15)', 'PAR(4)', cont='k_gp/k_pe:hb2', line_style_unstable='solid', ignore=['UZ'],
                           ax=ax5, line_color_stable='#148F77', line_color_unstable='#148F77')
ax5.set_xlabel(r"$k_{gp}$")
ax5.set_ylabel(r"$k_{pe}$")
ax5.set_xlim([0.0, 25.0])
ax5.set_ylim([0.0, 25.0])
ax5.set_title('C')

# psd and firing rates
rates = beta_data['results']
psds = beta_data['psds']

ax6 = fig.add_subplot(grid[4, :4])
ax6.plot(rates.index[100000:105000], rates.loc[10.0:10.49999, 'r_e'])
ax6.plot(rates.index[100000:105000], rates.loc[10.0:10.49999, 'r_p'])
ax6.set_ylabel('r')
ax6.set_title('E')
ax6.set_ylim([10.0, 100.0])

ax7 = fig.add_subplot(grid[4, 4:6])
ax7.plot(psds['freq_stn'][0], psds['pow_stn'][0])
ax7.plot(psds['freq_gpe'][0], psds['pow_gpe'][0])
ax7.set_ylabel('PSD')
ax7.set_in_layout(False)
ax7.set_xlim([0.0, 120.0])
ax7.set_xticks([0, 50, 100])

# 1D bifurcation diagram
ax8 = fig.add_subplot(grid[4, 6:])
ax8 = a2.plot_continuation('PAR(15)', 'U(3)', cont='k_gp:1', line_style_unstable='solid', ignore=['UZ'],
                           ax=ax8)
ax8 = a2.plot_continuation('PAR(15)', 'U(3)', cont='k_gp:1:lc1', line_style_unstable='solid', ignore=['UZ'],
                           ax=ax8, line_color_stable='#148F77', line_color_unstable='#148F77')
ax8.set_xlabel(r"$k_{gp}$")
ax8.set_ylabel(r"$r$")
ax8.set_xlim([0.0, 8.0])
ax8.set_ylim([0.0, 0.12])
ax8.set_xticks([0, 2, 4, 6, 8])
ax8.set_yticklabels(['0', '50', '100'])

# final touches
###############

# changes of figure layout
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)

# save figure
fig.canvas.draw()
plt.savefig(f'results/{fname}_beta.svg')

plt.show()
