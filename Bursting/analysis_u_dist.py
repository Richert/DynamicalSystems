import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from typing import Callable

# global variables
v_peak = 200.0
v_reset = -200.0


# function definitions
def v_dot(t: float, v: float, C: float, k: float, v_r: float, v_t: float, eta: float) -> float:
    return (k*(v-v_r)*(v-v_t) + eta) / C


def spike(t: float, v: np.ndarray, *args) -> float:
    event = v[0] - v_peak
    return event


def u_dot(t: float, u: float, v: Callable, b: float, v_r: float, tau_u: float) -> float:
    return (b*(v(t) - v_r) - u) / tau_u


def u_dist(u0: float, v0: float, t_init: float, T: float, dt: float, C: float, k: float, v_r: float, v_t: float, eta: float,
           b: float, tau_u: float, n_bins: int = 100, v_range: tuple = None, interpolation="cubic", **kwargs) -> dict:

    # integrate v
    t_eval = kwargs.pop("t_eval", np.linspace(0.0, T, num=int(T/dt)))
    t0 = 0.0
    spike.terminal = True
    v, time = [], []
    while t0 < T:
        res_v = solve_ivp(v_dot, (t0, T), np.asarray([v0]), args=(C, k, v_r, v_t, eta), events=spike,
                          t_eval=t_eval[t_eval >= t0], **kwargs)
        v_tmp = np.squeeze(res_v.y)
        t0 = res_v.t[-1]
        v0 = v_reset if len(res_v.y_events) > 0 else v_tmp[-1]
        v.extend(v_tmp[:-1].tolist())
        time.extend([t for t in res_v.t[:-1]])
    time.append(t0)
    v.append(v0)
    time = np.asarray(time)
    v = np.asarray(v)
    v_func = interp1d(time, v, kind=interpolation)

    # integrate u
    res_u = solve_ivp(u_dot, (0.0, T), np.asarray([u0]), args=(v_func, b, v_r, tau_u), t_eval=t_eval, **kwargs)
    u = np.squeeze(res_u.y)

    # remove initial condition
    v = v[time >= t_init]
    u = u[time >= t_init]
    time = time[time >= t_init]

    # estimate distribution of v
    v_hist, v_edges = np.histogram(v, bins=n_bins, range=v_range)

    # estimate distribution of the input to u
    inp = b*(v-v_r)
    inp_hist, inp_edges = np.histogram(inp, bins=n_bins)

    # estimate distribution of u
    u_hist, u_edges = np.histogram(u, bins=n_bins)

    res = {"v_hist": v_hist, "u_hist": u_hist, "inp_hist": inp_hist, "v": v, "u": u, "time": time,
           "v_edges": [(v_edges[i] + v_edges[i+1])/2 for i in range(len(v_edges)-1)],
           "u_edges": [(u_edges[i] + u_edges[i+1])/2 for i in range(len(u_edges)-1)],
           "inp_edges": [(inp_edges[i] + inp_edges[i+1])/2 for i in range(len(inp_edges)-1)],
           }
    return res


# define model parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
eta = 100.0
b = -1.0
tau_u = 50.0

# define initial state
v0 = -60.0
u0 = 0.0

# define simulation parameters
T = 1200.0
t0 = 200.0
dt = 0.01
method = "RK23"
atol = 1e-9
rtol = 1e-9

# define histogram parameters
n_bins = 500

# get results
res = u_dist(u0, v0, t0, T,  dt, C=C, k=k, v_r=v_r, v_t=v_t, eta=eta, b=b, tau_u=tau_u, n_bins=n_bins,
             v_range=(v_reset, v_peak), method=method, atol=atol, rtol=rtol)

# plot results
fig, axes = plt.subplots(nrows=4, figsize=(12, 10))

ax = axes[0]
v_edges, v_hist = res["v_edges"], res["v_hist"]
ax.bar(v_edges, v_hist / np.sum(v_hist), width=1.0)
ax.set_xlabel("v (mV)")
ax.set_ylabel("p")

ax = axes[1]
inp_edges, inp_hist = res["inp_edges"], res["inp_hist"]
ax.bar(inp_edges, inp_hist / np.sum(inp_hist), width=1.0)
ax.set_xlabel("b(v-v_r)")
ax.set_ylabel("p")

ax = axes[2]
u_edges, u_hist = res["u_edges"], res["u_hist"]
ax.bar(u_edges, u_hist / np.sum(u_hist), width=1.0)
ax.set_xlabel("u (pA)")
ax.set_ylabel("p")

ax = axes[3]
v, u, time = res["v"], res["u"], res["time"]
ax.plot(time, v, color="royalblue")
ax2 = ax.twinx()
ax2.plot(time, u, color="darkorange")
ax.set_xlabel("time (ms)")
ax.set_ylabel("v (mV)", color="royalblue")
ax2.set_ylabel("u (pA)", color="darkorange")

plt.tight_layout()
plt.show()
