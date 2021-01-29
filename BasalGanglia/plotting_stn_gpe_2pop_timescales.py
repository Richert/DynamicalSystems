from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyrates.utility.pyauto import PyAuto
import sys
sys.path.append('../')

# preparations
##############

fname = 'stn_gpe_2pop_timescales'

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/{fname}_conts.pkl", auto_dir=auto_dir)

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (5.25, 4.0)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['axes.titlepad'] = 0.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 25
cbar_shrink = 0.25
vmin = 10
vmax = 100

############
# plotting #
############

data = a.additional_attributes

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=1, ncols=3, figure=fig)

# 2d: tau_e x tau_p
ax1 = fig.add_subplot(grid[0, 0])
df1 = pd.DataFrame(np.ma.masked_where(data['tau_e_p_periods'] == 0, data['tau_e_p_periods']),
                   columns=data['tau_e'], index=data['tau_p'])
xmin1, xmax1 = np.min(data['tau_e']), np.max(data['tau_e'])
ymin1, ymax1 = np.min(data['tau_p']), np.max(data['tau_p'])
im1 = ax1.imshow(df1,
                 extent=(xmin1, xmax1, ymin1, ymax1),
                 origin='lower',
                 aspect=(xmax1-xmin1)/(ymax1-ymin1),
                 vmin=vmin, vmax=vmax
                 )
ax1 = a.plot_continuation('PAR(19)', 'PAR(20)', cont='tau_e/tau_p:hb1', line_style_unstable='solid', ignore=['UZ'],
                          ax=ax1, line_color_stable='#ee2b2b')
ax1.set_xlim([xmin1, xmax1])
ax1.set_ylim([ymin1, ymax1])
ax1.set_xlabel(r'$\tau_e$')
ax1.set_ylabel(r'$\tau_p$')
# fig.colorbar(im1, ax=ax1, shrink=cbar_shrink)

# 2d: tau_ampa x tau_gabaa
ax2 = fig.add_subplot(grid[0, 1])
df2 = pd.DataFrame(np.ma.masked_where(data['tau_ampa_gabaa_periods'] == 0, data['tau_ampa_gabaa_periods']),
                   columns=data['tau_ampa'], index=data['tau_gabaa'])
xmin2, xmax2 = np.min(data['tau_ampa']), np.max(data['tau_ampa'])
ymin2, ymax2 = np.min(data['tau_gabaa']), np.max(data['tau_gabaa'])
im2 = ax2.imshow(df2,
                 extent=(xmin2, xmax2, ymin2, ymax2),
                 origin='lower',
                 aspect=(xmax2 - xmin2) / (ymax2 - ymin2),
                 vmin=vmin, vmax=vmax
                 )
ax2 = a.plot_continuation('PAR(21)', 'PAR(22)', cont='tau_ampa_d/tau_gabaa_d:hb1', line_style_unstable='solid',
                          ignore=['UZ'], ax=ax2, line_color_stable='#ee2b2b')
ax2.set_xlim([xmin2, xmax2])
ax2.set_ylim([ymin2, ymax2])
ax2.set_xlabel(r'$\tau_{\mathrm{ampa}}$')
ax2.set_ylabel(r'$\tau_{\mathrm{gabaa}}$')
# fig.colorbar(im2, ax=ax2, shrink=cbar_shrink)

# 2d: tau_ampa x tau_gabaa
ax3 = fig.add_subplot(grid[0, 2])
df3 = pd.DataFrame(np.ma.masked_where(data['tau_stn_gabaa_periods'] == 0, data['tau_stn_gabaa_periods']),
                   columns=data['tau_stn'], index=data['tau_gabaa2'])
xmin3, xmax3 = np.min(data['tau_stn']), np.max(data['tau_stn'])
ymin3, ymax3 = np.min(data['tau_gabaa2']), np.max(data['tau_gabaa2'])
im3 = ax3.imshow(df3,
                 extent=(xmin3, xmax3, ymin3, ymax3),
                 origin='lower',
                 aspect=(xmax3 - xmin3)/(ymax3 - ymin3),
                 vmin=vmin, vmax=vmax
                 )
ax3 = a.plot_continuation('PAR(18)', 'PAR(22)', cont='tau_gabaa_d/tau_stn:hb1', line_style_unstable='solid',
                          ignore=['UZ'], ax=ax3, line_color_stable='#ee2b2b')
ax3.set_xlim([xmin3, xmax3])
ax3.set_ylim([ymin3, ymax3])
ax3.set_xlabel(r'$\frac{\tau_{\mathrm{gabaa}}^e}{\tau_{\mathrm{gabaa}}^p}$')
ax3.set_ylabel(r'$\tau_{\mathrm{gabaa}}$')

# changes of figure layout
fig.colorbar(im3, ax=ax3, shrink=cbar_shrink)
fig.suptitle('Limit cycle boundaries and frequencies')
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)

plt.show()
