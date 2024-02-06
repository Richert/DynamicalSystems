import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def u_dot(t: float, u: float, v: float, b: float, v_r: float, tau_u: float) -> float:
    return (b*(v - v_r) - u) / tau_u


def u_dist(u0: float, T: float, v: float, b: float, v_r: float, tau_u: float, n_bins: int = 100, u_range: tuple = None,
           **kwargs) -> tuple:

    # integrate u
    res = solve_ivp(u_dot, (0.0, T), np.asarray([u0]), args=(v, b, v_r, tau_u), **kwargs)
    u = np.squeeze(res.y)

    # estimate distribution of u
    u_hist, u_edges = np.histogram(u, bins=n_bins, range=u_range)

    return u_hist, [(u_edges[i] + u_edges[i+1])/2 for i in range(len(u_edges)-1)], u, res.t


# define model parameters
v = -40.0
v_r = -60.0
b = 2.0
tau_u = 30.0
u0 = -50.0

# define simulation parameters
T = 10.0
t_eval = np.linspace(0, T, num=10000)
method = "RK23"
atol = 1e-12
rtol = 1e-12
max_step = 0.05

# define histogram parameters
n_bins = 100
u_min = -100.0
u_max = 100.0

# get results
u_hist, u_edges, u, time = u_dist(u0, T, v, b, v_r, tau_u, n_bins=n_bins, u_range=(u_min, u_max),
                                  method=method, atol=atol, rtol=rtol, max_step=max_step, t_eval=t_eval)

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 6))

ax = axes[0]
ax.bar(u_edges, u_hist / np.sum(u_hist), width=1.0)
ax.set_xlabel("u (pA)")
ax.set_ylabel("p")

ax = axes[1]
ax.plot(time, u)
ax.set_xlabel("time (ms)")
ax.set_ylabel("u (pA)")

plt.tight_layout()
plt.show()
