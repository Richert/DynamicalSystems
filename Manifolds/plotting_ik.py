from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyauto import PyAuto
import sys
sys.path.append('../')
import pickle
import numpy as np
from pandas import read_pickle

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/ik_bifs.pkl", auto_dir=auto_dir)

# load simulation data
fre = pickle.load(open("results/fre_results.p", "rb"))
snn = pickle.load(open(f"results/rnn_results.p", "rb"))

# load manifold deviation data
mf_data = read_pickle("results/manifold_deviations.p")

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 7)
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

# 2D continuation
#################

# 2D bifurcation diagram in I and D
ax = fig.add_subplot(grid[0, :3])
a.plot_continuation('PAR(5)', 'PAR(4)', cont=f'g/eta:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(5)', 'PAR(4)', cont=f'g/eta:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(5)', 'PAR(4)', cont=f'g/eta:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77')
a.plot_continuation('PAR(5)', 'PAR(4)', cont=f'g/eta:hb2', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77')
a.plot_continuation('PAR(5)', 'PAR(4)', cont=f'g/eta:hc1', ax=ax, line_style_stable='dotted')
ax.set_xlabel(r'$\bar\eta$ (pA)')
ax.set_ylabel(r'$g$ (nS)')
ax.set_title('(A) 2D bifurcation diagram')
ax.set_xlim([0.0, 100.0])
ax.set_ylim([0.0, 30.0])

# 1D continuations
##################

# plot continuation in input current for different deltas
ax = fig.add_subplot(grid[0, 3:])
a.plot_continuation('PAR(5)', 'U(2)', cont=f'eta:1', ax=ax, line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(5)', 'U(2)', cont=f'eta:1:lc', ax=ax, ignore=['BP'], line_color_stable='#148F77')
for eta in snn['etas']:
    ax.axvline(x=eta, color='blue', linestyle='--')
ax.set_xlabel(r'$\bar\eta$ (pA)')
ax.set_ylabel(r'$v$ (mV)')
ax.set_title(r'(B) 1D bifurcation diagram for $g = 15$')
ax.set_ylim([-60.0, -36.0])
ax.set_xlim([30.0, 80.0])

# time series
#############

for i, (eta, snn_res) in enumerate(zip(snn['etas'], snn['results'])):

    # extract relevant signals
    snn_signal = np.squeeze(snn_res['v'])
    idx = np.argmin(np.abs(fre['map'].loc['eta', :] - eta))
    fre_signal = fre['results'].loc[:, fre['map'].columns.values[idx]]

    # plot signals
    ax = fig.add_subplot(grid[1, i*2:(i+1)*2])
    ax.plot(fre_signal.index, snn_signal)
    ax.plot(fre_signal)
    ax.set_xlabel('time (ms)')
    ax.set_ylim([-57.0, -38.0])
    ax.set_title(rf'$\bar \eta = {eta}$')
    if i == 0:
        ax.set_ylabel('v (mV)')
        plt.legend(['SNN', 'FRE'])

# manifold deviation stuff
##########################

# plot deviation matrix
in_var_label = r"var(I_{ext})"
D = mf_data['D']
ax = fig.add_subplot(grid[2, :2])
ax.imshow(D, cmap='copper')
ax.set_yticks(np.arange(0, D.shape[0]), labels=D.index)
ax.set_xticks(np.arange(0, D.shape[1]), labels=D.columns.values)
ax.set_ylabel(r'$p$')
ax.set_xlabel(rf"${in_var_label}$")

# plot exemplary input distributions in network
dists = mf_data['dists']
cols = ['blue', 'orange', 'green']
linestyles = ['solid', 'dashed']
lb, ub = -60.0, 60.0
n_bins = 50
ax1 = fig.add_subplot(grid[2, 2:4])
ax2 = fig.add_subplot(grid[2, 4:])
lines, legends = [], []
for l, (p, in_var_data) in enumerate(dists.items()):

    # draw input distribution
    if p > 0.2:
        labels = []
        for i, (in_var, data) in enumerate(in_var_data.items()):
            idx = (data['dist'] >= lb) * (data['dist'] <= ub)
            ax1.hist(data['dist'][idx], bins=n_bins, density=True, histtype='step', color=cols[i])
            labels.append(in_var)
        plt.sca(ax1)
        plt.legend(labels, title=rf"${in_var_label}$")
        ax1.set_xlabel(r'$\bar \eta + I_{ext}$')
        ax1.set_ylabel('PDF')
        ax1.set_xlim([lb, ub])

    # plot exemplary local mean-field distributions
    for i, (in_var, data) in enumerate(in_var_data.items()):
        d = data['dist'][data['s1']]
        idx = (d >= lb) * (d <= ub)
        _, _, line = ax2.hist(d[idx], bins=n_bins, density=True, histtype='step', linestyle=linestyles[l],
                              color=cols[i])
        lines.append(line[0])
        legends.append(rf"$p = {p}$, ${in_var_label} = {in_var}$")
plt.sca(ax2)
plt.legend(lines, legends)
ax2.set_xlim([-60, 60])
ax2.set_xlabel(r'$\bar \eta + I_{ext}$')
ax2.set_ylabel('PDF')

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/ik_bifs.pdf')
plt.show()
