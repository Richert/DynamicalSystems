import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from pyauto import PyAuto

sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/eic.pkl", auto_dir=auto_dir)
a0 = PyAuto.from_file(f"results/eic_corrected0.pkl", auto_dir=auto_dir)
a1 = PyAuto.from_file(f"results/eic_corrected1.pkl", auto_dir=auto_dir)
a2 = PyAuto.from_file(f"results/eic_corrected2.pkl", auto_dir=auto_dir)
deltas = a.additional_attributes['deltas']

# load simulation data
rnn_bfs = pickle.load(open("results/eic_results.p", "rb"))
rnn2_bfs = pickle.load(open("results/eic_results2.p", "rb"))
rnn3_bfs = pickle.load(open("results/eic_results3.p", "rb"))
fre_correct = pickle.load(open("results/eic_fre_corrected.p", "rb"))['results']
fre_uncorrect = pickle.load(open("results/eic_fre_uncorrected.p", "rb"))['results']
rnn_correct = pickle.load(open("results/eic_rnn_correct.p", "rb"))['results']
rnn_uncorrect = pickle.load(open("results/eic_rnn_uncorrect.p", "rb"))['results']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
s = 20


############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=4, ncols=6, figure=fig)

# 2D continuations
##################

n_points = 2

# continuation in uncorrected model
ax = fig.add_subplot(grid[:2, :2])
ax.scatter(rnn_bfs['lp1'][::n_points, 0], rnn_bfs['lp1'][::n_points, 1], c='#5D6D7E', marker='x', s=s, alpha=0.5)
ax.scatter(rnn_bfs['lp2'][::n_points, 0], rnn_bfs['lp2'][::n_points, 1], c='#5D6D7E', marker='x', s=s, alpha=0.5)
ax.scatter(rnn_bfs['hb1'][::n_points, 0], rnn_bfs['hb1'][::n_points, 1], c='#148F77', marker='x', s=s, alpha=0.5)
ax.scatter(rnn_bfs['hb2'][::n_points, 0], rnn_bfs['hb2'][::n_points, 1], c='#148F77', marker='x', s=s, alpha=0.5)
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_stable='dashed', line_style_unstable='dashed')
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_stable='dashed', line_style_unstable='dashed')
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_stable='dashed', line_style_unstable='dashed')
line1 = a0.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                             line_color_unstable='#5D6D7E', line_style_unstable='solid')
line2 = a0.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                             line_color_unstable='#5D6D7E', line_style_unstable='solid')
l1 = line1.get_paths()[0].vertices
l2_tmp = line2.get_paths()[0].vertices
l2 = np.interp(l1[:, 1], l2_tmp[:, 1], l2_tmp[:, 0])
plt.fill_betweenx(y=l1[:, 1], x2=l1[:, 0], x1=l2, color='#5D6D7E', alpha=0.5)
line = a0.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                            line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5)
ax.set_ylabel(r'$\Delta_{fs}$')
ax.set_xlabel(r'$I_{fs}$')
ax.set_title(r'(A) $v_{p} = 1000$, $v_0 = -1000$')
ax.set_ylim([0.0, 1.6])
ax.set_xlim([10.0, 80.0])

# continuation in corrected model
ax = fig.add_subplot(grid[:2, 2:4])
ax.scatter(rnn2_bfs['lp1'][::n_points, 0], rnn2_bfs['lp1'][::n_points, 1], c='#5D6D7E', marker='x', s=s, alpha=0.5)
ax.scatter(rnn2_bfs['lp2'][::n_points, 0], rnn2_bfs['lp2'][::n_points, 1], c='#5D6D7E', marker='x', s=s, alpha=0.5)
ax.scatter(rnn2_bfs['hb1'][::n_points, 0], rnn2_bfs['hb1'][::n_points, 1], c='#148F77', marker='x', s=s, alpha=0.5)
ax.scatter(rnn2_bfs['hb2'][::n_points, 0], rnn2_bfs['hb2'][::n_points, 1], c='#148F77', marker='x', s=s, alpha=0.5)
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_stable='dashed', line_style_unstable='dashed')
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_stable='dashed', line_style_unstable='dashed')
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_stable='dashed', line_style_unstable='dashed')
line1 = a1.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                             line_color_unstable='#5D6D7E', line_style_unstable='solid')
line2 = a1.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                             line_color_unstable='#5D6D7E', line_style_unstable='solid')
l1 = line1.get_paths()[0].vertices
l2_tmp = line2.get_paths()[0].vertices
l2 = np.interp(l1[:, 1], l2_tmp[:, 1], l2_tmp[:, 0])
plt.fill_betweenx(y=l1[:, 1], x2=l1[:, 0], x1=l2, color='#5D6D7E', alpha=0.5)
line = a1.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                            line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5, edgecolor="none")
ax.set_ylabel(r'$\Delta_{fs}$')
ax.set_xlabel(r'$I_{fs}$')
ax.set_title(r'(B) $v_{p} = 50$, $v_0 = -100$')
ax.set_ylim([0.0, 1.6])
ax.set_xlim([10.0, 80.0])

# continuation in corrected model in case of v_0 = v_r
ax = fig.add_subplot(grid[:2, 4:6])
ax.scatter(rnn3_bfs['lp1'][::n_points, 0], rnn3_bfs['lp1'][::n_points, 1], c='#5D6D7E', marker='x', s=s, alpha=0.5)
ax.scatter(rnn3_bfs['lp2'][::n_points, 0], rnn3_bfs['lp2'][::n_points, 1], c='#5D6D7E', marker='x', s=s, alpha=0.5)
ax.scatter(rnn3_bfs['hb1'][::n_points, 0], rnn3_bfs['hb1'][::n_points, 1], c='#148F77', marker='x', s=s, alpha=0.5)
ax.scatter(rnn3_bfs['hb2'][::n_points, 0], rnn3_bfs['hb2'][::n_points, 1], c='#148F77', marker='x', s=s, alpha=0.5)
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_stable='dashed', line_style_unstable='dashed')
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_stable='dashed', line_style_unstable='dashed')
a.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_stable='dashed', line_style_unstable='dashed')
line1 = a2.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp1', ax=ax, line_color_stable='#5D6D7E',
                             line_color_unstable='#5D6D7E', line_style_unstable='solid')
line2 = a2.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:lp2', ax=ax, line_color_stable='#5D6D7E',
                             line_color_unstable='#5D6D7E', line_style_unstable='solid')
l1 = line1.get_paths()[0].vertices
l2_tmp = line2.get_paths()[0].vertices
l2 = np.interp(l1[:, 1], l2_tmp[:, 1], l2_tmp[:, 0])
plt.fill_betweenx(y=l1[:, 1], x2=l1[:, 0], x1=l2, color='#5D6D7E', alpha=0.5)
line = a2.plot_continuation('PAR(36)', 'PAR(30)', cont=f'D_fs/I_fs:hb1', ax=ax, line_color_stable='#148F77',
                            line_color_unstable='#148F77', line_style_unstable='solid')
line_data = line.get_paths()[0].vertices
plt.fill_between(x=line_data[:, 0], y1=np.zeros_like(line_data[:, 0]), y2=line_data[:, 1], color='#148F77', alpha=0.5, edgecolor="none")
ax.set_ylabel(r'$\Delta_{fs}$')
ax.set_xlabel(r'$I_{fs}$')
ax.set_title(r'(C) $v_{p} = 40$, $v_0 = -60$')
ax.set_ylim([0.0, 1.6])
ax.set_xlim([10.0, 80.0])

# time series
#############

data = [(fre_uncorrect, rnn_uncorrect), (fre_correct, rnn_correct)]
time = np.linspace(1000, 4000, num=30000)
titles = [r'(D) Two-population model dynamics for $v_{p} = 1000$, $v_0 = -1000$',
          r'(E) Two-population model dynamics for $v_{p} = 50$, $v_0 = -100$']
for i, ((fre, rnn), title) in enumerate(zip(data, titles)):
    ax = fig.add_subplot(grid[2+i, :])
    l1 = ax.plot(time, rnn["se"], color="blue")
    l2 = ax.plot(time, rnn["si"], color="orange")
    l3 = ax.plot(time, fre["se"], color="blue", linestyle='dashed')
    l4 = ax.plot(time, fre["si"], color="orange", linestyle='dashed')
    xmin = np.min(rnn["si"])
    xmax = np.max(rnn["si"])
    plt.fill_betweenx([xmin-0.1*xmax, xmax+0.1*xmax], x1=1500, x2=2500.0, color='grey', alpha=0.15)
    plt.fill_betweenx([xmin-0.1*xmax, xmax+0.1*xmax], x1=2500, x2=3500.0, color='grey', alpha=0.3)
    ax.set_ylim([xmin-0.1*xmax, xmax+0.1*xmax])
    ax.set_title(title)
    if i == len(data)-1:
        ax.set_xlabel('time (ms)')
    elif i == 0:
        plt.legend([l1[0], l2[0], l3[0], l4[0]], [r"$s_{rs}$ (spiking network)", r"$s_{fs}$ (spiking network)",
                                                  r"$s_{rs}$ (mean-field)", r"$s_{fs}$ (mean-field)"], loc=2)
    ax.set_ylabel(r'$r$')

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/eic_comparison.pdf')
plt.show()
