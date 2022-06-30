from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyauto import PyAuto
import sys
sys.path.append('../')
import pickle
import numpy as np

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/ik_bifs.pkl", auto_dir=auto_dir)

# load simulation data
fre = pickle.load(open("results/fre_results.p", "rb"))
snn = pickle.load(open(f"results/rnn_results.p", "rb"))

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 5)
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
grid = gridspec.GridSpec(nrows=2, ncols=6, figure=fig)

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

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/ik_bifs.pdf')
plt.show()
