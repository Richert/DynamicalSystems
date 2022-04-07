from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from pyauto import PyAuto
import sys
import pickle
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/eic2.pkl", auto_dir=auto_dir)

# load simulation data
# fre_hom = pickle.load(open(f"results/eic_fre_hom.p", "rb"))['results']
# fre_het = pickle.load(open(f"results/eic_fre_het.p", "rb"))['results']

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

# 2D continuations
##################

# continuation of Hopf in Delta_ib and I_ib
ax = fig.add_subplot(grid[0:2, 0])
a.plot_continuation('PAR(7)', 'PAR(18)', cont=f'D_rs/I_rs:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77', line_style_unstable='solid')
a.plot_continuation('PAR(7)', 'PAR(18)', cont=f'D_rs/I_rs:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(7)', 'PAR(18)', cont=f'D_rs/I_rs:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
ax.set_xlabel(r'$\Delta_{rs}$')
ax.set_ylabel(r'$I_{rs}$')
ax.set_title('(A) 2D bifurcation diagram')
# ax.set_xlim([0.0, 3.0])
# ax.set_ylim([100.0, 400.0])

# 1D continuations
##################

# 1D continuation in I_ib for Delta_ib = 1.0
ax = fig.add_subplot(grid[0, 1])
a.plot_continuation('PAR(18)', 'U(1)', cont='I_rs:1', ax=ax, line_color_stable='#76448A', line_color_unstable='#5D6D7E')
ax.set_ylabel(r'$v_{rs}$')
ax.set_title(r'(B) 1D bifurcation diagram for $\Delta = 1.0$')
# ax.set_xlim([150.0, 350.0])

# 1D continuation in I_ib for Delta_ib = 2.0
ax = fig.add_subplot(grid[1, 1])
a.plot_continuation('PAR(18)', 'U(1)', cont='I_rs:2', ax=ax, line_color_stable='#76448A', line_color_unstable='#5D6D7E')
ax.set_xlabel(r'$I_{rs}$')
ax.set_ylabel(r'$v_{rs}$')
ax.set_title(r'(C) 1D bifurcation diagram for $\Delta = 2.0$')
# ax.set_xlim([150.0, 350.0])

# time series
#############

titles = [r'(D) $\Delta = 1.0$', r'(E) $\Delta = 2.0$', ]
# data = [[fre_hom], [fre_het]]
# for i, (title, (fre,)) in enumerate(zip(titles, data)):
#
#     ax = fig.add_subplot(grid[2, i])
#     ax.plot(fre['rs'])
#     ax.plot(fre['ib'])
#     ax.set_xlabel(r'time (ms)')
#     ax.set_title(title)
#     if i == 0:
#         plt.legend(['FRE', 'RNN'])
#         ax.set_ylabel(r'$v$')

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/eic2.pdf')
plt.show()
