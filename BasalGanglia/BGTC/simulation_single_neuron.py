import numpy as np
import matplotlib.pyplot as plt

# parameter definitions
#######################

# constants
v_cutoff = 1000.0
v_reset = -1000.0
tau_r = 50.0

# neuron parameters
C = 100.0
k = 1.0
v_r = -60.0
v_t = -40.0
a = 0.02
b = -10.0
d = 0.8
eta = 100.0
tau_x = 250.0

# function definitions
######################


def ik_run(T, dt, C, k, eta, v_r, v_t, a, b, d, tau_x, tau_r):

    v = v_r
    u = 0.0
    x = 0.0
    r = 0.0
    steps = int(T/dt)
    v_hist = []
    r_hist = []
    for step in range(steps):
        dv = (k * (v - v_r) * (v - v_t) + eta - u) / C
        du = a * (b * (v - v_r) - u) + x
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
        v_hist.append(v)
        r_hist.append(r)

    return v_hist, r_hist


# simulation
############

# simulation parameters
T = 2000.0
dt = 1e-2
time = np.arange(int(T/dt))*dt

# simulation
vs, rates = ik_run(T, dt, C, k, eta, v_r, v_t, a, b, d, tau_x, tau_r)

# plotting
##########

fig, axes = plt.subplots(nrows=2, figsize=(12, 6))
ax = axes[0]
ax.plot(time, vs)
ax.set_xlabel("time (ms)")
ax.set_ylabel("v (mV)")
ax = axes[1]
ax.plot(time, rates)
ax.set_xlabel("time (ms)")
ax.set_ylabel("r (Hz)")
plt.tight_layout()
plt.show()
