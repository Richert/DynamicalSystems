from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from pycobi import ODESystem
import sys
import numpy as np
import pickle
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a_1pop = ODESystem.from_file(f"results/rs.pkl", auto_dir=auto_dir)
a_2pop = ODESystem.from_file(f"results/eic.pkl", auto_dir=auto_dir)
a_3pop = ODESystem.from_file(f"results/eiic.pkl", auto_dir=auto_dir)

# load simulation data
fre_2pop_hom = pickle.load(open(f"results/eic_fre_hom.p", "rb"))['results']
fre_2pop_het = pickle.load(open(f"results/eic_fre_het.p", "rb"))['results']
fre_3pop_hom = pickle.load(open(f"results/eiic_fre_hom.p", "rb"))['results']
fre_3pop_het = pickle.load(open(f"results/eiic_fre_het2.p", "rb"))['results']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 5.5)
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
grid = gridspec.GridSpec(nrows=4, ncols=6, figure=fig)

# bifurcation diagrams
######################

# RS
deltas_1pop = a_1pop.additional_attributes["deltas"]
n = len(deltas_1pop)
cmap = plt.get_cmap('copper', lut=n)
ax = fig.add_subplot(grid[:2, :2])
lines = []
for j in range(1, n + 1):
    c = to_hex(cmap(j, alpha=1.0))
    line = a_1pop.plot_continuation('PAR(16)', 'U(1)', cont=f'I:{j}', ax=ax, line_color_stable=c, line_color_unstable=c)
    lines.append(line)
ax.set_xlim([0.0, 70.0])
ax.set_yticks([0.0, 0.01, 0.02, 0.03, 0.04])
ax.set_yticklabels(['0', '10', '20', '30', '40'])
ax.set_xlabel(r'$I$ (pA)')
ax.set_ylabel(r'$r$ (Hz)')
ax.set_title(r'(A) regular spiking neurons')
plt.legend(handles=lines, labels=[D for D in deltas_1pop], title=r"$\Delta_v$ (mV)", loc=1)

# RS-FS
deltas_2pop = a_2pop.additional_attributes["deltas"]
ax = fig.add_subplot(grid[:2, 2:4])
line1 = a_2pop.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', line_style_unstable='solid')
line2 = a_2pop.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', line_style_unstable='solid')
l1 = line1.get_paths()[0].vertices
l2_tmp = line2.get_paths()[0].vertices
l2 = np.interp(l1[:, 1], l2_tmp[:, 1], l2_tmp[:, 0])
plt.fill_betweenx(y=l1[:, 1], x2=l1[:, 0], x1=l2, color='#5D6D7E', alpha=0.5)
line3 = a_2pop.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                                 line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line3.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
line4 = a_2pop.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:pd1', ax=ax, line_style_unstable='solid',
                                 ignore=['LP'], line_color_stable='#4287f5', line_color_unstable='#4287f5')
line_data = line4.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#4287f5', alpha=0.5)
ax.axhline(y=deltas_2pop[1], color='black', linestyle='--')
plt.text(25.0, deltas_2pop[1]+0.02, "D")
ax.axhline(y=deltas_2pop[2], color='black', linestyle='--')
plt.text(25.0, deltas_2pop[2]+0.02, "E")
points = [c for c in ax.collections if c.get_offsets().data.shape[0] == 1]
gh, cp = points[5], points[0]
ax.set_ylabel(r'$\Delta_{fs}$ (mV)')
ax.set_xlabel(r'$I_{fs}$ (pA)')
ax.set_title('(B) rs+fs, $\Delta_{rs} = 1.0$ mV')
ax.set_ylim([0.0, 1.0])
ax.set_xlim([20.0, 80.0])

# RS-FS-LTS
delta_str = r"\Delta_{fs}"
p1 = 'PAR(54)'
p2 = 'PAR(48)'
u = 'U(1)'
neuron = 'lts'
deltas_3pop = a_3pop.additional_attributes["deltas"]
ax = fig.add_subplot(grid[:2, 4:])
line1 = a_3pop.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:1:lp1', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', line_style_unstable='solid')
line2 = a_3pop.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:1:lp2', ax=ax, line_color_stable='#5D6D7E',
                                 line_color_unstable='#5D6D7E', line_style_unstable='solid')
l1 = line1.get_paths()[0].vertices
l2_tmp = line2.get_paths()[0].vertices
l2 = np.interp(l1[:, 1], l2_tmp[:, 1], l2_tmp[:, 0])
plt.fill_betweenx(y=l1[:, 1], x2=l1[:, 0], x1=l2, color='#5D6D7E', alpha=0.5)
line = a_3pop.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:1:hb1', ax=ax, line_color_stable='#148F77',
                                line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
line = a_3pop.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:1:hb2', ax=ax, line_color_stable='#148F77',
                                line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
line = a_3pop.plot_continuation(p1, p2, cont=f'D_{neuron}/I_{neuron}:1:pd1', ax=ax, line_style_unstable='solid',
                                ignore=['LP'], line_color_stable='#4287f5', line_color_unstable='#4287f5')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#4287f5', alpha=0.5)
ax.axhline(y=0.1, color='black', linestyle='--')
plt.text(75.0, 0.12, "F")
ax.axhline(y=0.6, color='grey', linestyle='--')
plt.text(75.0, 0.62, "G")
ax.set_ylabel(r'$\Delta_{lts}$ (mV)')
ax.set_xlabel(r'$I_{lts}$ (pA)')
ax.set_title(fr'(C) rs+fs+lts, ${delta_str} = {deltas_3pop[0]}$ mV, ' + r'$\Delta_{rs} = 1.0$')
ax.set_ylim([0.0, 2.0])
ax.set_xlim([70.0, 140.0])
plt.legend([line1, line3, line4] + [gh, cp],
           ["Fold", "Andronov-Hopf", "Period Doubling", "Generalized Hopf", "Cusp"],
           loc=1)

# time series
#############

# RS-FS
data = [fre_2pop_hom, fre_2pop_het]
titles = [fr'(D) ${delta_str} = {deltas_2pop[1]}$ mV',
          fr'(E) ${delta_str} = {deltas_2pop[2]}$ mV']
for i, (fre, title) in enumerate(zip(data, titles)):
    ax = fig.add_subplot(grid[2, i*3:(i+1)*3])
    ax.plot(fre)
    xmin = np.min(fre.values)
    xmax = np.max(fre.values)
    plt.fill_betweenx([xmin-0.1*xmax, xmax+0.1*xmax], x1=2000, x2=2500.0, color='grey', alpha=0.15)
    plt.fill_betweenx([xmin-0.1*xmax, xmax+0.1*xmax], x1=2500, x2=3000.0, color='grey', alpha=0.3)
    ax.set_xlabel('time (ms)')
    ax.set_ylim([xmin-0.1*xmax, xmax+0.1*xmax])
    ax.set_title(title)
    ax.set_ylabel(r'$r$ (Hz)')
    plt.legend(fre.columns.values, loc=2)
    if i == len(data)-1:
        ax.set_yticks([0.0, 0.025, 0.05])
        ax.set_yticklabels(['0', '25', '50'])
    elif i == 0:
        ax.set_yticks([0.0, 0.1, 0.2])
        ax.set_yticklabels(['0', '100', '200'])

# RS-FS-LTS
data = [fre_3pop_hom, fre_3pop_het]
titles = [r'(F) $\Delta_{lts} = 0.1$ mV',
          r'(G) $\Delta_{lts} = 0.6$ mV']
time = fre_3pop_hom.index
for i, (fre, title) in enumerate(zip(data, titles)):
    ax = fig.add_subplot(grid[3, i*3:(i+1)*3])
    ax.plot(time, fre)
    xmin = np.min(fre.values)
    xmax = np.max(fre.values)
    plt.fill_betweenx([xmin - 0.1 * xmax, xmax + 0.1 * xmax], x1=1500, x2=2500.0, color='grey', alpha=0.15)
    plt.fill_betweenx([xmin - 0.1 * xmax, xmax + 0.1 * xmax], x1=2500, x2=3500.0, color='grey', alpha=0.3)
    ax.set_xlabel('time (ms)')
    ax.set_ylim([xmin-0.1*xmax, xmax+0.1*xmax])
    ax.set_title(title)
    ax.set_ylabel(r'$r$ (Hz)')
    if i == 0:
        ax.set_yticks([0.0, 0.07, 0.14])
        ax.set_yticklabels(['0', '70', '140'])
    elif i == len(data)-1:
        ax.set_yticks([0.0, 0.06, 0.12])
        ax.set_yticklabels(['0', '60', '120'])
    plt.legend(fre.columns.values, loc=2)

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/ik_combined.pdf')
plt.show()
