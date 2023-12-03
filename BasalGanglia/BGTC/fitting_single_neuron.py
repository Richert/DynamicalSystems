import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# define parameters
###################

# parameters to train
C = [40.0, 80.0]
k = [0.1, 10.0]
eta = [0.0, 100.0]
v_r = [-60.0, -50.0]
v_t = [-55.0, -40.0]
a = [0.0, 0.5]
b = [-10.0, 10.0]
d = [0.0, 200.0]

# constants
v_cutoff = 1000.0
v_reset = -1000.0

# function definitions
######################


def ik_run(T, dt, I_ext, C, k, eta, v_r, v_t, a, b, d):

    v = v_r
    u = 0.0
    steps = int(T/dt)
    spikes = 0
    for step in range(steps):
        dv = (k * (v - v_r) * (v - v_t) + eta + I_ext - u) / C
        du = a * (b * (v - v_r) - u)
        v += dt * dv
        u += dt * du
        if v > v_cutoff:
            v = v_reset
            u += d
            spikes += 1

    return spikes*1e3/T


# perform optimization
######################

# inputs and target rates
inputs = [25.0, 75.0, 100.0, 125.0, 150.0, 175.0]
target_rates = [25.0, 50.0, 70.0, 100.0, 160.0, 200.0]

# simulation parameters
T = 1000.0
dt = 1e-2


# rate-calculating function
def get_rates(inputs, *theta):
    return np.asarray([ik_run(T, dt, I_ext, *theta) for I_ext in inputs])


# loss function
def loss(theta, target_rates, inputs):
    rates = get_rates(inputs, *theta)
    return np.sum([(r-t)**2 for r, t in zip(rates, target_rates)])


# initial parameter vector
params = [C, k, eta, v_r, v_t, a, b, d]
theta = np.asarray([np.mean(p) for p in params])

# optimization
res = minimize(loss, theta, bounds=params, args=(target_rates, inputs), method="Nelder-Mead", tol=1e-8,
               options={"disp": True, "maxiter": 10000})

# final parameters
param_names = ["C", "k", "eta", "v_r", "v_t", "a", "b", "d"]
fitted_params = res.x
print(f"Final parameters: {[f'{key} = {val}' for key, val in zip(param_names, fitted_params)]}")

# get rates for final parameters
rates = get_rates(inputs, *fitted_params)

# plotting
##########

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(inputs, target_rates, label="targets")
ax.plot(inputs, rates, label="fit")
ax.set_xlabel("input (pA)")
ax.set_ylabel("spike rates (Hz)")
ax.set_title("Model fit")
ax.legend()
plt.tight_layout()
plt.show()
