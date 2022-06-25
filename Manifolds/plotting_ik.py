from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyauto import PyAuto
import sys
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/ik_bifs.pkl", auto_dir=auto_dir)

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 4)
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
grid = gridspec.GridSpec(nrows=2, ncols=1, figure=fig)

# 2D continuation
#################

# 2D bifurcation diagram in I and D
ax = fig.add_subplot(grid[0, 0])
a.plot_continuation('PAR(5)', 'PAR(4)', cont=f'g/eta:lp1', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(5)', 'PAR(4)', cont=f'g/eta:lp2', ax=ax, line_color_stable='#5D6D7E',
                    line_color_unstable='#5D6D7E', line_style_unstable='solid')
a.plot_continuation('PAR(5)', 'PAR(4)', cont=f'g/eta:hb1', ax=ax, line_color_stable='#148F77',
                    line_color_unstable='#148F77')
ax.set_xlabel(r'$\bar\eta$')
ax.set_ylabel(r'$g$')
ax.set_title('(A) 2D bifurcation diagram')
ax.set_xlim([0.0, 100.0])
ax.set_ylim([0.0, 30.0])

# 1D continuations
##################

# plot continuation in input current for different deltas
ax = fig.add_subplot(grid[1, 0])
a.plot_continuation('PAR(5)', 'U(1)', cont=f'eta:1', ax=ax, line_color_unstable='#5D6D7E')
a.plot_continuation('PAR(5)', 'U(1)', cont=f'eta:1:lc', ax=ax, ignore=['BP'], line_color_stable='#148F77')
ax.set_xlabel(r'$\bar\eta$')
ax.set_ylabel(r'$r$')
ax.set_title(r'(B) 1D bifurcation diagram for $g = 15$')
# ax.set_ylim([0.0, 0.9])
ax.set_xlim([0.0, 100.0])

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/ik_bifs.pdf')
plt.show()
