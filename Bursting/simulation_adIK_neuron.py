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
d = 2.0
eta = 100.0
tau_x = 300.0

# constants
v_cutoff = 100.0
v_reset = -100.0

# function definitions
######################


def adik_run(T, dt, C, k, eta, v_r, v_t, tau_u, b, d, tau_x):

    v = v_r
    u = 0.0
    x = 0.0

    v_hist = []
    u_hist = []
    x_hist = []

    steps = int(T / dt)
    for step in range(steps):

        dv = (k * (v - v_r) * (v - v_t) + eta - u) / C
        du = (b * (v - v_r) - u) / tau_u + d*x
        dx = -x/tau_x

        v += dt * dv
        u += dt * du
        x += dt * dx

        if v > v_cutoff:
            v = v_reset
            x += 1.0

        v_hist.append(v)
        u_hist.append(u)
        x_hist.append(x)

    return v_hist, u_hist, x_hist


def adik_nullclines(v, k, eta, v_r, v_t, tau_u, b, d, x):

    nc1 = k*v**2 + k*(-v_r-v_t)*v + k*v_r*v_t + eta
    nc2 = b*(v-v_r) + tau_u*d*x

    return nc1, nc2


# simulation
############

# simulation parameters
T = 2000.0
dt = 1e-3
time = np.arange(int(T/dt))*dt
margin = 20.0
cutoff = int(500.0/dt)

results = {"b": bs, "x": [], "v": [], "u": [], "nc": []}
for b in bs:

    # simulation
    vs, us, xs = adik_run(T, dt, C, k, eta, v_r, v_t, tau_u, b, d, tau_x)

    # nullcline calculation
    v = np.linspace(v_r-margin, v_t+margin, num=1000)
    x_vals = np.asarray([np.min(xs[cutoff:]), np.max(xs[cutoff:])])
    nullclines = []
    for x in x_vals:
        nullclines.append(adik_nullclines(v, k, eta, v_r, v_t, tau_u, b, d, x))

    # collect results
    results["v"].append(vs)
    results["u"].append(us)
    results["x"].append(xs)
    results["nc"].append(nullclines)

# plotting
##########

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
markersize = 6

# figure layout
fig = plt.figure()
grid = fig.add_gridspec(nrows=4, ncols=6)

# simulated time series
for i, (b, v, u, x, ncs) in enumerate(zip(results["b"], results["v"], results["u"], results["x"], results["nc"])):

    # membrane potential dynamics
    ax = fig.add_subplot(grid[0, i*2:(i+1)*2])
    ax.plot(time, v)
    ax.set_ylabel(r"$v$ (mv)")
    ax.set_title(fr"$b = {b}$")

    # recovery variable
    ax = fig.add_subplot(grid[1, i * 2:(i + 1) * 2])
    ax.plot(time, u, color="darkorange")
    ax.set_ylabel(r"$u$ (pA)")

    # spike frequency adaptation
    ax = fig.add_subplot(grid[2, i * 2:(i + 1) * 2])
    ax.plot(time, x)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$x$ (dimensionless)")

    # nullclines
    x_vals = np.asarray([np.min(x[cutoff:]), np.max(x[cutoff:])])
    v = np.linspace(v_r - margin, v_t + margin, num=1000)
    for j, nc in enumerate(ncs):
        ax = fig.add_subplot(grid[3, i*2+j])
        ax.plot(v, nc[0], label=r"$\dot v = 0$")
        ax.plot(v, nc[1], label=r"$\dot u = 0$")
        ax.set_xlabel(r"$v$ (mV)")
        if j == 0:
            ax.set_ylabel(r"$u$ (pA)")
        ax.set_ylim([0.0, 400.0])
        ax.set_xlim([-75.0, -25.0])
        ax.set_title(rf"$x = {np.round(x_vals[j], decimals=1)}$")

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/adik_dynamics.pdf')
plt.show()
