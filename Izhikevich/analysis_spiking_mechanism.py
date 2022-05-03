import numpy as np
import matplotlib.pyplot as plt

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 3)
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
v_reset = [-60.0, -70.0, -80.0, -100.0, -120.0, -140.0]
v_spike = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
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

# plotting
fig, ax = plt.subplots(ncols=3)
for fr, diff, inp_mf in zip(frs, diffs, inp_mfs):
    ax[0].plot(inp, fr*1e3)
    ax[1].plot(inp, diff*1e3)
    ax[2].plot(inp, inp_mf)
plt.sca(ax[2])
plt.legend([fr'${v}$' for v in v_reset], loc=2, title=r'$\mathrm{v_0}$')
ax[0].set_xlabel(r'$I$')
ax[1].set_xlabel(r'$I$')
ax[2].set_xlabel(r'$I$')
ax[0].set_ylabel(r'$r_i$')
ax[1].set_ylabel(r'$r_i-r_{\infty}$')
ax[2].set_ylabel(r'$I^*$')
ax[0].set_title('(A) Input-output curves')
ax[1].set_title('(B) Output differences')
ax[2].set_title('(C) Input differences')
plt.tight_layout()

# saving
plt.savefig(f'results/spiking_mechanism.pdf')
plt.show()
