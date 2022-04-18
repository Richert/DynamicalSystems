from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from pyauto import PyAuto
import sys
import pickle
sys.path.append('../')

# choose neuron type
ntype = 'ib'

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/{ntype}_fre_io.pkl", auto_dir=auto_dir)

# load simulation data
rnn = pickle.load(open(f"results/{ntype}_rnn_io.p", "rb"))

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (7, 3)
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
ax = fig.add_subplot()

# plot IO curves
a.plot_continuation('PAR(16)', 'U(4)', cont=f'I:1', ax=ax)
ax.plot(rnn['inputs'], rnn['results'], c='orange', marker='*')
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$s$')
ax.set_xlim([0.0, 500.0])
ax.set_title('Population input-output relationship')
plt.legend(['FRE', 'RNN'])

# finishing touches
###################

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/{ntype}_io.pdf')
plt.show()
