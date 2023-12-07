import numpy as np
import matplotlib.pyplot as plt

# parameter definitions
#######################

# neuron parameters
C = 100.0
k = 0.7
v_r = -60.0
v_t = -40.0
tau_u = 100.0
b = -2.0
d = 100.0
eta = 200.0
tau_x = 200.0

# constants
v_cutoff = 100.0
v_reset = -100.0
u_cutoff = 1000.0
u_reset = -1000.0
tau_r = 50.0

# function definitions
######################


def ik_run(T, dt, C, k, eta, v_r, v_t, tau_u, b, d, tau_x, tau_r):

    v = v_r
    u = 0.1
    x = 0.0
    r = 0.0
    steps = int(T/dt)
    v_hist = []
    r_hist = []
    u_hist = []
    for step in range(steps):
        dv = (k * (v - v_r) * (v - v_t) + eta - u - x) / C
        du = (b*(v-v_r) - u)*x/(tau_u*tau_x)
        # du = (b * (v - v_r) - u) / tau_u + x
        dx = -x/tau_x
        dr = -r/tau_r
        v += dt * dv
        u += dt * du
        x += dt * dx
        r += dt * dr
        if v > v_cutoff:
            v = v_reset
            x += d
            r += 1.0/dt
        if u > u_cutoff:
            u = u_reset
        v_hist.append(v)
        r_hist.append(r)
        u_hist.append(u)

    return v_hist, r_hist, u_hist


# simulation
############

# simulation parameters
T = 2000.0
dt = 1e-3
time = np.arange(int(T/dt))*dt

# simulation
vs, rates, us = ik_run(T, dt, C, k, eta, v_r, v_t, tau_u, b, d, tau_x, tau_r)

# plotting
##########

fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
ax = axes[0]
ax.plot(time, vs)
ax.set_xlabel("time (ms)")
ax.set_ylabel("v (mV)")
ax = axes[1]
ax.plot(time, rates)
ax.set_xlabel("time (ms)")
ax.set_ylabel("r (Hz)")
ax = axes[2]
ax.plot(time, us)
ax.set_xlabel("time (ms)")
ax.set_ylabel("u (?)")
plt.tight_layout()
plt.show()
