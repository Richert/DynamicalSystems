import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

def magnesium_block(v, mg, k, v0, E_r):
    return (E_r - v) / (1 + mg/k * np.exp(v/v0))

def polynomial(x, v):
    return x[0] + x[1]*v + (x[2] if len(x) > 2 else 0.0)*v**2

def loss(x, v, mg, k, v0, E_r):
    v_mg = magnesium_block(v, mg, k, v0, E_r)
    v_pn = polynomial(x, v)
    return np.abs(v_mg - v_pn)

def p2_approx(v, v0, v1, v2, c1, c2, sigma):
    exact = np.zeros_like(v)
    exact[(v >= v0) & (v < v1)] = c1
    exact[(v >= v1) & (v < v2)] = c2
    exact[v >= v2] = 0.0
    approx = (c1*(1 + np.tanh((v-v0)/sigma)) + (c2-c1)*(1 + np.tanh((v-v1)/sigma)) - c2*(1 + np.tanh((v-v2)/sigma)))/2
    return exact, approx

# parameters
N = 1000
v_thresholds = [-150.0, -30.0, 7.0, 20.0]
v_dists = np.asarray([v_thresholds[i+1]-v_thresholds[i] for i in range(3)]) / (v_thresholds[-1] - v_thresholds[0])
x0 = np.random.randn(3)
mg = 1.0
k = 3.57
v0 = -16.13
E_r = 0.0
sigma = 1.0

# optimization
results = {"v": [], "x": []}
for i in range(3):
    v_range = np.linspace(v_thresholds[i], v_thresholds[i+1], N)
    x1 = leastsq(loss, x0 if i < 2 else x0[:2], args=(v_range, mg, k, v0, E_r))[0]
    results["v"].append(v_range)
    results["x"].append(x1)

# plot results
fig, axes = plt.subplots(ncols=3, figsize=(12, 4),
                         # gridspec_kw={"width_ratios": v_dists}
                         )
for i, (v, x) in enumerate(zip(results["v"], results["x"])):
    ax = axes[i]
    ax.plot(v, magnesium_block(v, mg, k, v0, E_r), label="target", color="black")
    ax.plot(v, polynomial(x, v), label="fit", color="darkorange")
    if i < 2:
        ax.set_title(f"a = {np.round(x[0], decimals=4)}, b = {np.round(x[1], decimals=4)}, c = {np.round(x[2], decimals=4)}")
    else:
        ax.set_title(f"a = {np.round(x[0], decimals=4)}, b = {np.round(x[1], decimals=4)}")
    ax.legend()
    ax.set_ylim([-20.0, 15.0])
    ax.set_xlabel("v")
    if i == 0:
        ax.set_ylabel("f(v)")
plt.tight_layout()

fig, ax = plt.subplots(figsize=(12, 4))
v = np.linspace(v_thresholds[0], v_thresholds[-1], N)
p2t, p2f = p2_approx(v, v_thresholds[0], v_thresholds[1], v_thresholds[2], results["x"][0][-1], results["x"][1][-1], sigma)
ax.plot(v, p2t, label="target")
ax.plot(v, p2f, label="fit")
ax.set_xlabel("v")
ax.set_ylabel("p2")
ax.legend()
plt.tight_layout()

plt.show()
