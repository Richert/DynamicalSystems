import matplotlib.pyplot as plt
import numpy as np


def coordinates(x) -> tuple:
    return np.real(x), np.imag(x)


# generate data
thetas = np.asarray([-3.1, -0.5, -1.0, 0.0, 0.2, 3.1])
thetas_2d = np.exp(1.0j*thetas)
kmo = np.mean(thetas_2d)
avg_phase = np.exp(1.0j*np.mean(thetas))

# plot data
fig, ax = plt.subplots(figsize=(10, 10))
circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
ax.add_patch(circ)
x, y = coordinates(thetas_2d)
ax.scatter(x, y, marker="o", color="blue")
kmo_x, kmo_y = coordinates(kmo)
ax.plot([0.0, kmo_x], [0.0, kmo_y], color="red")
avg_x, avg_y = coordinates(avg_phase)
ax.plot([0.0, avg_x], [0.0, avg_y], color="green")
ax.set_xlabel("real")
ax.set_ylabel("imag")
ax.set_title("Unit Circle")
plt.tight_layout()
plt.show()
