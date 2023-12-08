import numpy as np
import matplotlib.pyplot as plt

# parameter definitions
#######################

# neuron parameters
tau_u = 100.0
b = 1.0
x = 100.0
v_r = -60.0
v = 100.0

# constants
u_cutoff = 1000.0
u_reset = -1000.0
tau_r = 50.0

# function definitions
######################


def u_run(T, dt, tau_u, b, x, v, v_r, tau_r):

    u = 0.0
    w = 0.0
    r = 0.0

    steps = int(T/dt)
    w_hist = []
    u_hist = []
    r_hist = []

    for step in range(steps):
        du = (w-u)/tau_u
        dw = (b*(v-v_r)-w)*(x - w)/tau_u
        dr = -r/tau_r
        u += dt * du
        w += dt * dw
        r += dt * dr
        if u > u_cutoff:
            u = u_reset
            w = u_reset
            r += 1.0/dt
        u_hist.append(u)
        w_hist.append(w)
        r_hist.append(r)

    return u_hist, w_hist, r_hist


# simulation
############

# simulation parameters
T = 500.0
dt = 1e-3
time = np.arange(int(T/dt))*dt

# simulation
us, ws, rates = u_run(T, dt, tau_u, b, x, v, v_r, tau_r)

# plotting
##########

fig, axes = plt.subplots(nrows=3, figsize=(12, 6))
ax = axes[0]
ax.plot(time, us)
ax.set_xlabel("time (ms)")
ax.set_ylabel("u (mV)")
ax = axes[1]
ax.plot(time, ws)
ax.set_xlabel("time (ms)")
ax.set_ylabel("w (mV)")
ax = axes[2]
ax.plot(time, rates)
ax.set_xlabel("time (ms)")
ax.set_ylabel("r (Hz)")
plt.tight_layout()
plt.show()
