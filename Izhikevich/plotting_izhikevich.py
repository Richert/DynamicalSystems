from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyauto import PyAuto
import sys
sys.path.append('../')

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/izhikevich_fre.pkl", auto_dir=auto_dir)
gs = a.additional_attributes['g']

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# plot continuation in resting membrane potential for different synaptic coupling strengths
ax = fig.add_subplot(grid[0, 0])
n = len(gs)
lines = []
for i in range(1, n+1):
    ax = a.plot_continuation('PAR(16)', 'U(1)', cont=f'I:{i}', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E')
    lines.append(ax)
# ax.set_xlim([-12.0, -2.0])
# ax.set_ylim([0., 1.2])
ax.set_xlabel(r'$I$')
ax.set_ylabel('Firing rate (r)')
ax.set_title('Fixed Points')
plt.legend(handles=lines, labels=[f'g = {g}' for g in gs])

# plot eta continuation for single alpha with limit cycle continuation
target = a.additional_attributes['target'] + 1
ax = fig.add_subplot(grid[0, 1])
ax = a.plot_continuation('PAR(16)', 'U(1)', cont=f'I:{target}', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#5D6D7E')
ax = a.plot_continuation('PAR(16)', 'U(1)', cont=f'I:{target}:lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                         custom_bf_styles={'LP': {'marker': 'p'}})
# ax.set_xlim([-6.5, -3.8])
# ax.set_ylim([0., 2.5])
ax.set_xlabel(r'$I$')
ax.set_ylabel('Firing rate (r)')
ax.set_title(r'Limit Cycle')

# 2D bifurcation diagram I
ax = fig.add_subplot(grid[1, 0])
ax = a.plot_continuation('PAR(16)', 'PAR(4)', cont=f'g/I:hb1', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#76448A')
ax = a.plot_continuation('PAR(16)', 'PAR(4)', cont=f'g/I:lp1', ax=ax, line_color_stable='#5D6D7E',
                         line_color_unstable='#5D6D7E')
ax = a.plot_continuation('PAR(16)', 'PAR(4)', cont=f'g/I:lp2', ax=ax, line_color_stable='#5D6D7E',
                         line_color_unstable='#5D6D7E')
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$g$')

# 2D bifurcation diagram II
ax = fig.add_subplot(grid[1, 1])
ax = a.plot_continuation('PAR(16)', 'PAR(6)', cont=f'D/I:hb1', ax=ax, line_color_stable='#76448A',
                         line_color_unstable='#76448A')
ax = a.plot_continuation('PAR(16)', 'PAR(6)', cont=f'D/I:lp1', ax=ax, line_color_stable='#5D6D7E',
                         line_color_unstable='#5D6D7E')
ax = a.plot_continuation('PAR(16)', 'PAR(6)', cont=f'D/I:lp2', ax=ax, line_color_stable='#5D6D7E',
                         line_color_unstable='#5D6D7E')
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$\Delta$')

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.show()
