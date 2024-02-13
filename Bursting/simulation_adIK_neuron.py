import numpy as np
import matplotlib.pyplot as plt

# parameter definitions
#######################

# neuron parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
tau_u = 35.0
bs = [-20.0, -10.0, 0.0]
d = 400.0
eta = 100.0
tau_x = 300.0

# constants
v_cutoff = 100.0
v_reset = -100.0

# function definitions
######################


def adik_vf(v, u, x, C, k, eta, v_r, v_t, tau_u, b, d, tau_x):
    dv = (k * (v - v_r) * (v - v_t) + eta - u) / C
    du = (b * (v - v_r) - u) / tau_u + d * x
    dx = -x / tau_x
    return np.asarray([dv, du, dx])


def adik_run(T, dt, C, k, eta, v_r, v_t, tau_u, b, d, tau_x):

    v = v_r
    u = 0.0
    x = 0.0

    v_hist = []
    u_hist = []
    x_hist = []

    steps = int(T / dt)
    for step in range(steps):

        dy = adik_vf(v, u, x, C, k, eta, v_r, v_t, tau_u, b, d, tau_x)
        v += dt * dy[0]
        u += dt * dy[1]
        x += dt * dy[2]

        if v > v_cutoff:
            v = v_reset
            x += 1.0/tau_x

        v_hist.append(v)
        u_hist.append(u)
        x_hist.append(x)

    return v_hist, u_hist, x_hist


def adik_nullclines(v, k, eta, v_r, v_t, tau_u, b, d, x):

    nc1 = k*v**2 + k*(-v_r-v_t)*v + k*v_r*v_t + eta
    nc2 = b*(v-v_r) + tau_u*d*x

    return nc1, nc2


def norm(vals):
    return np.sqrt(np.sum([val**2 for val in vals]))


# simulation
############

# simulation parameters
T = 2000.0
dt = 1e-3
time = np.arange(int(T/dt))*dt
margin = 20.0
cutoff = int(500.0/dt)

# vectorfield grid
vgrid = np.linspace(-70, -30, num=4)
ugrid = np.linspace(40, 360, num=5)

results = {"b": bs, "x": [], "v": [], "u": [], "nc": [], "vf": []}
for b in bs:

    # simulation
    vs, us, xs = adik_run(T, dt, C, k, eta, v_r, v_t, tau_u, b, d, tau_x)

    # nullcline and vectorfield calculation
    v = np.linspace(v_r-margin, v_t+margin, num=1000)
    x_vals = np.asarray([np.min(xs[cutoff:]), np.max(xs[cutoff:])])
    nullclines, vfs = [], []
    for x in x_vals:
        nullclines.append(adik_nullclines(v, k, eta, v_r, v_t, tau_u, b, d, x))
        vfs.append(
            np.asarray([[adik_vf(v, u, x, C, k, eta, v_r, v_t, tau_u, b, d, tau_x)[:2] for v in vgrid] for u in ugrid])
        )

    # collect results
    results["v"].append(vs)
    results["u"].append(us)
    results["x"].append(xs)
    results["nc"].append(nullclines)
    results["vf"].append(vfs)

# plotting
##########

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# vectorfield settings
vec_norm = 1.5

# figure layout
fig = plt.figure(layout="constrained")
fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0.005, wspace=0.005)
grid = fig.add_gridspec(nrows=4, ncols=6)

# simulated time series
titles = ["A", "B", "C"]
for i, (b, v, u, x, ncs, vfs) in (
        enumerate(zip(results["b"], results["v"], results["u"], results["x"], results["nc"], results["vf"]))):

    # membrane potential dynamics
    ax = fig.add_subplot(grid[0, i*2:(i+1)*2])
    ax.plot(time, v)
    if i == 0:
        ax.set_ylabel(r"$v$ (mv)")
    ax.set_title(fr"({titles[i]}) $b = {b}$")

    # recovery variable
    ax = fig.add_subplot(grid[1, i * 2:(i + 1) * 2])
    ax.plot(time, u, color="darkorange")
    ax.set_ylim([-400.0, 450.0])
    if i == 0:
        ax.set_ylabel(r"$u$ (pA)")

    # spike frequency adaptation
    ax = fig.add_subplot(grid[2, i * 2:(i + 1) * 2])
    ax.plot(time, x)
    ax.set_ylim([0.0, 0.022])
    ax.set_xlabel("time (ms)")
    if i == 0:
        ax.set_ylabel(r"$a$ (1/ms)")

    # nullclines and vectorfields
    x_vals = np.asarray([np.min(x[cutoff:]), np.max(x[cutoff:])])
    v = np.linspace(v_r - margin, v_t + margin, num=1000)
    for j, (nc, vf) in enumerate(zip(ncs, vfs)):
        ax = fig.add_subplot(grid[3, i*2+j])
        ax.plot(v, nc[0], label=r"$\dot v = 0$")
        ax.plot(v, nc[1], label=r"$\dot u = 0$")
        if j == 0:
            ax.set_xlabel(r"$v$ (mV)")
        for row in range(vf.shape[0]):
            for col in range(vf.shape[1]):
                dv, du = vf[row, col, 0], vf[row, col, 1]
                scale = vec_norm
                ax.annotate("", xy=[vgrid[col] + scale*dv, ugrid[row] + scale*du],
                            xytext=[vgrid[col], ugrid[row]], arrowprops=dict(width=0.5, headlength=4.0, headwidth=3.0,
                                                                             shrink=0.01, edgecolor="none",
                                                                             facecolor="black"))
        if i == 0 and j == 0:
            ax.set_ylabel(r"$u$ (pA)")
        ax.set_ylim([0.0, 400.0])
        ax.set_xlim([-75.0, -25.0])
        if j > 0:
            ax.set_yticklabels([])
        ax.set_title(rf"$a = {np.round(x_vals[j], decimals=3)}$")

# saving/plotting
fig.canvas.draw()
grid.set_width_ratios([1, 1, 1, 1, 1, 1])
plt.savefig(f'results/adik_dynamics.pdf')
plt.show()
