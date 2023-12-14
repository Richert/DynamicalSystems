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
b = -20.0
d = 0.1
eta = 0.0
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


def adik_nullclines(v, k, eta, v_r, v_t, tau_u, b, x):

    nc1 = k*v**2 + k*(-v_r-v_t)*v + k*v_r*v_t + eta
    nc2 = b*(v-v_r) + tau_u*x

    return nc1, nc2


# simulation
############

# simulation parameters
T = 2000.0
dt = 1e-3
time = np.arange(int(T/dt))*dt

# simulation
vs, us, xs = adik_run(T, dt, C, k, eta, v_r, v_t, tau_u, b, d, tau_x)

# nullcline calculation
margin = 20.0
v = np.linspace(v_r-margin, v_t+margin, num=1000)
x_vals = np.asarray([0.0, 0.005, 0.02]) * tau_x
nullclines = []
for x in x_vals:
    nullclines.append(adik_nullclines(v, k, eta, v_r, v_t, tau_u, b, x))

# plotting
##########

# simulated time series
fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
ax = axes[0]
ax.plot(time, vs)
ax.set_xlabel("time (ms)")
ax.set_ylabel("v (mV)")
ax.set_title("adaptive IK neuron dynamics")
ax = axes[1]
ax.plot(time, us)
ax.set_xlabel("time (ms)")
ax.set_ylabel("u (pA)")
ax = axes[2]
ax.plot(time, xs)
ax.set_xlabel("time (ms)")
ax.set_ylabel("x")
plt.tight_layout()

# nullclines
_, axes = plt.subplots(ncols=3, figsize=(12, 4))
for ax, nc, x in zip(axes, nullclines, x_vals):
    ax.plot(v, nc[0], label=r"$\dot v = 0$")
    ax.plot(v, nc[1], label=r"$\dot u = 0$")
    ax.set_xlabel("v (mV)")
    ax.set_ylabel("u (pA)")
    ax.set_title(rf"ad-IK nullclines for $x = {x}$")
plt.tight_layout()
plt.show()
