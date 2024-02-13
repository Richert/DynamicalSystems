import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from typing import Callable

# global variables
v_peak = 200.0
v_reset = -200.0


# function definitions
######################

def v_dot(t: float, v: float, C: float, k: float, v_r: float, v_t: float, eta: float) -> float:
    return (k*(v-v_r)*(v-v_t) + eta) / C


def spike(t: float, v: np.ndarray, *args) -> float:
    event = v[0] - v_peak
    return event


def u_dot(t: float, u: float, v: Callable, x: Callable, b: float, v_r: float, tau_u: float, kappa: float) -> float:
    return (b*(v(t) - v_r) - u) / tau_u + kappa*x(t)


def adik_dot(t: float, y: np.ndarray, C: float, k: float, v_r: float, v_t: float, eta: float, b: float, tau_u: float,
             kappa: float, tau_x: float) -> np.ndarray:
    v, u, x = y[0], y[1], y[2]
    dv = (k*(v-v_r)*(v-v_t) + eta - u) / C
    du = (b*(v - v_r) - u) / tau_u + kappa*x
    dx = -x / tau_x
    return np.asarray([dv, du, dx])


def solve(u0: float, v0: float, t_init: float, T: float, dt: float, C: float, k: float, v_r: float, v_t: float,
          eta: float, b: float, tau_u: float, kappa: float, tau_x: float, n_bins: int = 100,
          interpolation="cubic", **kwargs) -> dict:

    # integrate v
    t_eval = kwargs.pop("t_eval", np.linspace(0.0, T, num=int(T/dt)))
    t0 = 0.0
    x0 = 0.0
    spike.terminal = True
    v, time, x = [], [], []
    while t0 < T:
        res_v = solve_ivp(v_dot, (t0, T), np.asarray([v0]), args=(C, k, v_r, v_t, eta), events=spike,
                          t_eval=t_eval[t_eval >= t0], **kwargs)
        v_tmp = np.squeeze(res_v.y)
        t0 = res_v.t[-1]
        v0 = v_reset if len(res_v.y_events) > 0 else v_tmp[-1]
        v.extend(v_tmp[:-1].tolist())
        x.extend(x0*np.exp(-res_v.t[:-1]/tau_x))
        x0 = 1.0/tau_x if len(res_v.y_events) > 0 else np.exp(-res_v.t[-1]/tau_x)
        time.extend([t for t in res_v.t[:-1]])
    time.append(t0)
    v.append(v0)
    x.append(x0)
    time = np.asarray(time)
    v = np.asarray(v)
    x = np.asarray(x)
    v_func = interp1d(time, v, kind=interpolation)
    x_func = interp1d(time, x, kind=interpolation)

    # integrate u
    res_u = solve_ivp(u_dot, (0.0, T), np.asarray([u0]), args=(v_func, x_func, b, v_r, tau_u, kappa),
                      t_eval=t_eval, **kwargs)
    u = np.squeeze(res_u.y)

    # integrate IK equations
    t0 = 0.0
    x0 = 0.0
    v2, u2 = [], []
    while t0 < T:
        res_ik = solve_ivp(adik_dot, (t0, T), np.asarray([v0, u0, x0]),
                           args=(C, k, v_r, v_t, eta, b, tau_u, kappa, tau_x),
                           events=spike, t_eval=t_eval[t_eval >= t0], **kwargs)
        v_tmp, u_tmp, x_tmp = res_ik.y[0, :], res_ik.y[1, :], res_ik.y[2, :]
        t0 = res_ik.t[-1]
        v0 = v_reset if len(res_ik.y_events) > 0 else v_tmp[-1]
        u0 = u_tmp[-1]
        x0 = x_tmp[-1] + len(res_ik.y_events)/tau_x
        v2.extend(v_tmp[:-1].tolist())
        u2.extend(u_tmp[:-1].tolist())
    v2.append(v0)
    u2.append(u0)
    v2 = np.asarray(v2)
    u2 = np.asarray(u2)

    # remove initial condition
    v = v[time >= t_init]
    u = u[time >= t_init]
    v2 = v2[time >= t_init]
    u2 = u2[time >= t_init]
    time = time[time >= t_init]

    # estimate distribution of v
    v_hist, v_edges = np.histogram(v, bins=n_bins)

    # estimate distribution of the input to u
    inp = b*(v-v_r)
    inp_hist, inp_edges = np.histogram(inp, bins=n_bins)

    # estimate distribution of u
    u_hist, u_edges = np.histogram(u, bins=n_bins)

    # estimate distributions of u2 and v2
    v2_hist, v2_edges = np.histogram(v2, bins=n_bins)
    u2_hist, u2_edges = np.histogram(u2, bins=n_bins)

    res = {"v_hist": v_hist, "u_hist": u_hist, "inp_hist": inp_hist, "v": v, "u": u, "time": time,
           "v_edges": [(v_edges[i] + v_edges[i+1])/2 for i in range(len(v_edges)-1)],
           "u_edges": [(u_edges[i] + u_edges[i+1])/2 for i in range(len(u_edges)-1)],
           "inp_edges": [(inp_edges[i] + inp_edges[i+1])/2 for i in range(len(inp_edges)-1)],
           "v2_hist": v2_hist, "u2_hist": u2_hist, "v2": v2, "u2": u2,
           "v2_edges": [(v2_edges[i] + v2_edges[i+1])/2 for i in range(len(v2_edges)-1)],
           "u2_edges": [(u2_edges[i] + u2_edges[i + 1]) / 2 for i in range(len(u2_edges) - 1)],
           }
    return res


# analysis
##########

# define model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
eta = 100.0
b = -15.0
tau_u = 35.0
tau_x = 300.0
kappas = [0.0, 400.0]

# define initial state
v0 = -60.0
u0 = 0.0

# define simulation parameters
T = 2500.0
t0 = 500.0
dt = 0.01
method = "RK23"
atol = 1e-9
rtol = 1e-9

# define histogram parameters
n_bins = 500

# get results
results = []
for kappa in kappas:
    res = solve(u0, v0, t0, T,  dt, C=C, k=k, v_r=v_r, v_t=v_t, eta=eta, b=b, tau_u=tau_u, kappa=kappa, tau_x=tau_x,
                n_bins=n_bins, method=method, atol=atol, rtol=rtol)
    results.append(res)

# plotting
##########

# plot settings
plt.rcParams['backend'] = "TkAgg"
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0

# create figure layout
fig = plt.figure(layout="constrained")
fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0.005, wspace=0.005)
subfigs = fig.subfigures(2, 2)
fig.suptitle("Lorentzian ansatz I: AdIK neuron statistics")

t_plot = [800.0, 800.0]
titles = [["(A) no feedback", r"(B) feedback from $a$"], [r"(C) feedback from $u$", "(D) feedback from $u$ and $a$"]]
for i, (kappa, res) in enumerate(zip(kappas, results)):

    # set up grid specs
    stop = int(t_plot[i] / dt)
    grid1 = subfigs[0, i].add_gridspec(2, 2)
    grid2 = subfigs[1, i].add_gridspec(2, 2)
    subfigs[0, i].suptitle(titles[0][i])
    subfigs[1, i].suptitle(titles[1][i])

    # set up axes
    ax1 = subfigs[0, i].add_subplot(grid1[0, 0])
    ax2 = subfigs[0, i].add_subplot(grid1[0, 1])
    ax3 = subfigs[0, i].add_subplot(grid1[1, :])
    ax4 = subfigs[1, i].add_subplot(grid2[0, 0])
    ax5 = subfigs[1, i].add_subplot(grid2[0, 1])
    ax6 = subfigs[1, i].add_subplot(grid2[1, :])

    # plot v distribution
    ax = ax1
    ax.set_yscale("log")
    # ax.set_ylim([0.0, 0.08])
    # ax.set_yticks([0.0, 0.04, 0.08])
    v_edges, v_hist = res["v_edges"], res["v_hist"]
    ax.bar(v_edges, v_hist / np.sum(v_hist), width=1.0, color="royalblue")
    ax.set_xlabel(r"$v$ (mV)")
    ax.set_ylabel(r"$\log(\rho)$")

    # plot u distribution
    ax = ax2
    ax.set_yscale("log")
    # ax.set_ylim([0.0, 0.08])
    # ax.set_yticks([0.0, 0.04, 0.08])
    u_edges, u_hist = res["u_edges"], res["u_hist"]
    ax.bar(u_edges, u_hist / np.sum(u_hist), width=1.0, color="darkorange")
    ax.set_xlabel(r"$u$ (pA)")
    ax.set_ylabel("")

    # plot v and u dynamics
    ax = ax3
    v, u, time = res["v"], res["u"], res["time"]
    ax.plot(time[:stop], v[:stop], color="royalblue")
    ax2 = ax.twinx()
    ax2.plot(time[:stop], u[:stop], color="darkorange")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$v$ (mV)", color="royalblue")
    ax2.set_ylabel(r"$u$ (pA)", color="darkorange")

    # plot v2 distribution
    ax = ax4
    ax.set_yscale("log")
    # ax.set_ylim([0.0, 0.08])
    # ax.set_yticks([0.0, 0.04, 0.08])
    v_edges, v_hist = res["v2_edges"], res["v2_hist"]
    ax.bar(v_edges, v_hist / np.sum(v_hist), width=1.0, color="royalblue")
    ax.set_xlabel(r"$v$ (mV)")
    ax.set_ylabel(r"$\log(\rho)$")

    # plot u distribution
    ax = ax5
    ax.set_yscale("log")
    # ax.set_ylim([0.0, 0.08])
    # ax.set_yticks([0.0, 0.04, 0.08])
    u_edges, u_hist = res["u_edges"], res["u_hist"]
    ax.bar(u_edges, u_hist / np.sum(u_hist), width=1.0, color="darkorange")
    ax.set_xlabel(r"$u$ (pA)")
    ax.set_ylabel("")

    # plot v and u dynamics
    ax = ax6
    v, u = res["v2"], res["u2"]
    ax.plot(time[:stop], v[:stop], color="royalblue")
    ax2 = ax.twinx()
    ax2.plot(time[:stop], u[:stop], color="darkorange")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"$v$ (mV)", color="royalblue")
    ax2.set_ylabel(r"$u$ (pA)", color="darkorange")

    grid1.set_width_ratios([1, 1])
    grid2.set_width_ratios([1, 1])

# saving/plotting
fig.canvas.draw()
plt.savefig(f'results/adik_distributions.pdf')
plt.show()
