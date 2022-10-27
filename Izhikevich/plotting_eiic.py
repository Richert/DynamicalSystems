from matplotlib import gridspec
import matplotlib.pyplot as plt
from pycobi import ODESystem
import sys
import numpy as np
import pickle
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = ODESystem.from_file(f"results/eiic.pkl", auto_dir=auto_dir)
deltas = a.additional_attributes['deltas']

# load simulation data
fre_hom = pickle.load(open(f"results/eiic_fre_hom.p", "rb"))['results']
fre_het = pickle.load(open(f"results/eiic_fre_het.p", "rb"))['results']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
# plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 4.5)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6


############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=3, ncols=6, figure=fig)

# define relevant variables
delta_str = r"\Delta_{fs}"
p1 = 'PAR(54)'
p2 = 'PAR(48)'
u = 'U(1)'
neuron = 'lts'

# 2D continuations
##################

# continuation in Delta_lts and I_lts for low Delta_fs
ax = fig.add_subplot(grid[:2, :2])
line1 = a.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:1:lp1', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable='solid')
line2 = a.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:1:lp2', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable='solid')
l1 = line1.get_paths()[0].vertices
l2_tmp = line2.get_paths()[0].vertices
l2 = np.interp(l1[:, 1], l2_tmp[:, 1], l2_tmp[:, 0])
plt.fill_betweenx(y=l1[:, 1], x2=l1[:, 0], x1=l2, color='#5D6D7E', alpha=0.5)
line = a.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:1:hb1', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
line = a.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:1:hb2', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
# line = a.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:1:pd1', ax=ax, line_style_unstable='solid',
#                            ignore=['LP'], line_color_stable='#4287f5', line_color_unstable='#4287f5')
# line_data = line.get_paths()[0].vertices
# plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#4287f5', alpha=0.5)
ax.axhline(y=0.1, color='black', linestyle='--')
ax.axhline(y=0.6, color='grey', linestyle='--')
ax.axhline(y=1.8, color='grey', linestyle='--')
ax.set_ylabel(r'$\Delta_{lts}$ (mV)')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_title(fr'(A) ${delta_str} = {deltas[0]}$')
ax.set_ylim([0.0, 2.0])
ax.set_xlim([70.0, 140.0])

# continuation in Delta_lts and I_lts for high Delta_fs
ax = fig.add_subplot(grid[:2, 2:4])
line1 = a.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:2:lp1', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable='solid')
line2 = a.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:2:lp2', ax=ax, line_color_stable='#5D6D7E',
                            line_color_unstable='#5D6D7E', line_style_unstable='solid')
l1 = line1.get_paths()[0].vertices
l2_tmp = line2.get_paths()[0].vertices
l2 = np.interp(l1[:, 1], l2_tmp[:, 1], l2_tmp[:, 0])
plt.fill_betweenx(y=l1[:, 1], x2=l1[:, 0], x1=l2, color='#5D6D7E', alpha=0.5)
line = a.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:2:hb1', ax=ax, line_color_stable='#148F77',
                           line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
line = a.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:2:pd1', ax=ax, line_style_unstable='solid',
                           ignore=['LP'], line_color_stable='#4287f5', line_color_unstable='#4287f5')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#4287f5', alpha=0.5)
ax.axhline(y=0.1, color='black', linestyle='--')
ax.axhline(y=0.6, color='grey', linestyle='--')
ax.axhline(y=1.8, color='grey', linestyle='--')
ax.set_ylabel(r'$\Delta_{lts}$ (mV)')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_title(fr'(B) ${delta_str} = {deltas[-1]}$')
ax.set_ylim([0.0, 2.0])
ax.set_xlim([70.0, 140.0])

# 1D continuations
##################

a.update_bifurcation_style(bf_type='PD', marker='*', color='k')

# continuation in I_fs for low Delta_lts
ax = fig.add_subplot(grid[0, 4:])
a.plot_continuation(p1, u, cont=f'I_{neuron}:2', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
a.plot_continuation(p1, u, cont=f'I_{neuron}:2:lc1', ax=ax, line_color_stable='#148F77', ignore=['BP', 'UZ'])
ax.axvline(x=80.0, color='black', linestyle='--')
ax.axvline(x=105.0, color='grey', alpha=0.15, linestyle='--')
ax.axvline(x=130.0, color='grey', alpha=0.3, linestyle='--')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_ylabel(r'$r_{lts}$ (Hz)')
ax.set_title(fr'(C) ${delta_str} = {deltas[-1]}$' + r', $\Delta_{lts} = 0.1$')
ax.set_xlim([70.0, 140.0])
ax.set_ylim([0.0, 0.046])
ax.set_yticks([0.0, 0.02, 0.04])
ax.set_yticklabels(['0', '20', '40'])

# continuation in I_lts for high Delta_fs
ax = fig.add_subplot(grid[1, 4:])
a.plot_continuation(p1, u, cont=f'I_{neuron}:1', ax=ax, line_color_stable='#76448A',
                    line_color_unstable='#5D6D7E')
a.plot_continuation(p1, u, cont=f'I_{neuron}:1:lc1', ax=ax, line_color_stable='#148F77', ignore=['BP', 'UZ'])
a.plot_continuation(p1, u, cont=f'I_{neuron}:1:lc2', ax=ax, line_color_stable='#148F77', ignore=['BP', 'UZ'])
ax.axvline(x=80.0, color='black', linestyle='--')
ax.axvline(x=105.0, color='grey', alpha=0.15, linestyle='--')
ax.axvline(x=130.0, color='grey', alpha=0.3, linestyle='--')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_ylabel(r'$r_{lts}$ (Hz)')
ax.set_title(fr'(D) ${delta_str} = {deltas[-2]}$' + r', $\Delta_{lts} = 0.1$')
ax.set_xlim([70.0, 140.0])
ax.set_ylim([0.0, 0.046])
ax.set_yticks([0.0, 0.02, 0.04])
ax.set_yticklabels(['0', '20', '40'])

# time series
#############

data = [fre_hom, fre_het]
titles = [r'(E) $\Delta_{fs} = 0.4$, $\Delta_{lts} = 0.1$', r'(F) $\Delta_{fs} = 0.8$, $\Delta_{lts} = 0.1$']
for i, (fre, title) in enumerate(zip(data, titles)):
    ax = fig.add_subplot(grid[2, i*3:(i+1)*3])
    fre = fre*1e3
    ax.plot(fre)
    xmin = np.min(fre.values)
    xmax = np.max(fre.values)
    plt.fill_betweenx([xmin - 0.1 * xmax, xmax + 0.1 * xmax], x1=1500, x2=2500.0, color='grey', alpha=0.15)
    plt.fill_betweenx([xmin - 0.1 * xmax, xmax + 0.1 * xmax], x1=2500, x2=3500.0, color='grey', alpha=0.3)
    ax.set_xlabel('time (ms)')
    ax.set_ylim([xmin-0.1*xmax, xmax+0.1*xmax])
    ax.set_title(title)
    if i == 0:
        plt.legend(fre.columns.values, loc=2)
        ax.set_ylabel(r'$r$ (Hz)')

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/eiic.pdf')
plt.show()
