import numpy as np
import matplotlib.pyplot as plt

# generate data
n = 2
m = 10000

eta_min = -100.0
eta_max = 100.0
etas = np.random.uniform(low=eta_min, high=eta_max, size=(n,))

delta_min = 0.1
delta_max = 10.0
deltas = np.random.uniform(low=delta_min, high=delta_max, size=(n,))

dists = [eta + Delta * np.tan(0.5*np.pi*(2*np.arange(1, m+1)-m-1)/(m+1)) for eta, Delta in zip(etas, deltas)]

# plotting
n_bins = 200
x_min = -200.0
x_max = 200.0
fig, ax = plt.subplots(figsize=(12, 4))
for eta_dist, eta, Delta in zip(dists, etas, deltas):
    ax.hist(eta_dist, n_bins, range=(x_min, x_max), label=rf"$\eta = {eta}$, $\Delta = {Delta}$")
ax.legend()
ax.set_xlabel(r"$\eta$")
ax.set_ylabel("#")
plt.tight_layout()
plt.show()
