import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib import gridspec
from pyauto import PyAuto
import sys
sys.path.append('../')
import pickle

# load pyauto data
path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"
a = PyAuto.from_file(f"results/rs_corrected.pkl", auto_dir=auto_dir)
a2 = PyAuto.from_file(f"results/rs_uncorrected.pkl", auto_dir=auto_dir)

# load simulation data
fre_low = pickle.load(open(f"results/spike_mech_fre.p", "rb"))['results']
fre_high = pickle.load(open(f"results/spike_mech_fre2.p", "rb"))['results']
fre_inf = pickle.load(open(f"results/spike_mech_fre_inf.p", "rb"))['results']
rnn_low = pickle.load(open(f"results/spike_mech_rnn.p", "rb"))['results']
rnn_high = pickle.load(open(f"results/spike_mech_rnn2.p", "rb"))['results']

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
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.handlelength'] = 1.0
markersize = 6


def get_fr(inp: np.ndarray, k: float, C: float, v_reset: float, v_spike: float, v_r: float, v_t: float):
    fr = np.zeros_like(inp)
    alpha = v_r+v_t
    mu = 4*(v_r*v_t + inp/k) - alpha**2
    idx = mu > 0
    mu_sqrt = np.sqrt(mu[idx])
    fr[idx] = k*mu_sqrt/(2*C*(np.arctan((2*v_spike-alpha)/mu_sqrt) - np.arctan((2*v_reset-alpha)/mu_sqrt)))
    return fr


def correct_input_taylor(inp: np.ndarray, v_thr: float):
    inp_sqrt = np.sqrt(inp)
    inp_atan = np.arctan(v_thr/inp_sqrt)
    return inp*(1 - (1 - 4*inp_atan/np.pi**2) / (1 + v_thr*inp_sqrt/((v_thr**2 + inp)*inp_atan)))


def correct_input(inp: np.ndarray, v_thr: float, inp_0: np.ndarray):
    inp = (4*inp*np.arctan(v_thr/np.sqrt(inp_0))**2)/np.pi**2
    return inp


def correct_input_rec(inp: np.ndarray, v_thr: float, epsilon: float = 1.0):
    idx = inp > 0.0
    inp_pos = inp[idx]
    inp_0 = correct_input_taylor(inp_pos, v_thr)
    diff = np.inf
    while np.abs(diff) > epsilon:
        inp_1 = correct_input(inp_pos, v_thr, inp_0)
        diff = np.mean(inp_1-inp_0)
        inp_0 = inp_1
    inp[idx] = inp_0
    return inp


def correct_input_mf(inp: np.ndarray, k: float, v_r: float, v_t: float, v_spike: float, v_reset: float, g: float, s: float,
                     E: float):
    alpha = v_r + v_t + g*s/k
    mu = 4*(v_r*v_t + (inp + g*s*E)/k) - alpha**2
    idx = mu > 0
    mu_sqrt = np.sqrt(mu[idx])
    inp[idx] = np.pi**2*k*mu[idx]/(4*(np.arctan((2*v_spike-alpha)/mu_sqrt) - np.arctan((2*v_reset-alpha)/mu_sqrt))**2)
    inp[idx] += k*alpha**2/4
    inp[idx] -= k*v_r*v_t + g*s*E
    return inp


# parameters
inp = np.linspace(0.0, 200.0, num=1000)
v_reset = [-60.0, -70.0, -80.0, -100.0, -120.0, -np.inf]
v_spike = [50.0, 50.0, 50.0, 50.0, 50.0, np.inf]
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0

frs, diffs, inp_mfs = [], [], []
fr_mf = get_fr(inp, k, C, v_spike=np.inf, v_reset=-np.inf, v_r=v_r, v_t=v_t)
for v_res, v_sp in zip(v_reset, v_spike):

    # calculate firing rate
    frs.append(get_fr(inp, k, C, v_spike=v_sp, v_reset=v_res, v_r=v_r, v_t=v_t))

    # calculate differences to mean-field firing rate
    diffs.append(frs[-1] - fr_mf)

    # calculate mean-field input
    inp_mfs.append(correct_input_mf(inp.copy(), k, v_r, v_t, v_sp, v_res, 0.0, 0.0, 0.0))

############
# plotting #
############

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=4, ncols=6, figure=fig)

ax1 = fig.add_subplot(grid[:2, :2])
ax2 = fig.add_subplot(grid[:2, 2:4])
ax3 = fig.add_subplot(grid[:2, 4:])
cmap2 = plt.get_cmap('copper', lut=len(diffs))
for i, (fr, diff, inp_mf) in enumerate(zip(frs, diffs, inp_mfs)):
    c = to_hex(cmap2(len(diffs)-i, alpha=1.0))
    ax1.plot(inp, fr*1e3, c=c)
    ax2.plot(inp, diff*1e3, c=c)
    ax3.plot(inp, inp_mf, c=c)
plt.sca(ax2)
plt.legend([fr'${v}$' for v in v_reset], loc=2, title=r'$\mathrm{v_0}$')
ax1.set_xlabel(r'$I$')
ax2.set_xlabel(r'$I$')
ax3.set_xlabel(r'$I$')
ax1.set_ylabel(r'$r_i$')
ax2.set_ylabel(r'$r_i-r_{\infty}$')
ax3.set_ylabel(r'$I^*$')
ax1.set_title('(A) Input-output curves')
ax2.set_title('(B) Output differences')
ax3.set_title('(C) Input differences')

# plot the 1D bifurcation diagrams
v_reset = a.additional_attributes['v_reset']
n = len(v_reset)
cmap = plt.get_cmap('copper', lut=n)
ax = fig.add_subplot(grid[2:, :3])
lines = []
for j in range(1, n + 1):
    c = to_hex(cmap(j, alpha=1.0))
    line = a.plot_continuation('PAR(16)', 'U(4)', cont=f'I:{j}', ax=ax, line_color_stable=c, line_color_unstable=c)
    lines.append(line)
line = a2.plot_continuation('PAR(16)', 'U(4)', cont=f'I:1', ax=ax)
lines.append(line)
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$s$')
ax.set_title(r'(D) 1D bifurcation diagrams for different $v_0$')
ax.set_xlim([0.0, 80.0])
plt.legend(handles=lines, labels=[fr'${v}$' for v in v_reset] + [r'$\infty$'], loc=2, title=r'$\mathrm{v_0}$')

# plot the time signals
data = [[fre_low, rnn_low, fre_inf], [fre_high, rnn_high, fre_inf]]
titles = [rf'(E) $v_0 = {v_reset[-1]}$', rf'(F) $v_0 = {v_reset[0]}$']
time = np.linspace(500.0, 4500.0, num=fre_low['s'].shape[0])
for i, ((fre, rnn, inf), title) in enumerate(zip(data, titles)):
    ax = fig.add_subplot(grid[i+2, 3:])
    ax.plot(time, rnn['s'])
    ax.plot(time, fre['s'])
    ax.plot(time, inf['s'], c='black', linestyle='--')
    xmin = np.min(rnn['s'])
    xmax = np.max(rnn['s'])
    plt.fill_betweenx([xmin - 0.1 * xmax, xmax + 0.1 * xmax], x1=1500, x2=2500.0, color='grey', alpha=0.15)
    plt.fill_betweenx([xmin - 0.1 * xmax, xmax + 0.1 * xmax], x1=2500, x2=3500.0, color='grey', alpha=0.3)
    ax.set_ylim([xmin - 0.1 * xmax, xmax + 0.1 * xmax])
    if i == len(titles) - 1:
        ax.set_xlabel('time (ms)')
        plt.legend(['spiking network', r'mean-field (corrected)', 'mean-field (uncorrected)'])
    ax.set_ylabel('s')
    ax.set_title(title)

# saving
plt.savefig(f'results/spiking_mechanism.pdf')
plt.show()
