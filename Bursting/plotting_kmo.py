import matplotlib.pyplot as plt
import numpy as np


def coordinates(x) -> tuple:
    return np.real(x), np.imag(x)


# generate data
vs = np.asarray([-40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
thetas = 2.0 * np.arctan(vs)
thetas_2d = np.exp(1.0j*thetas)
kmo = np.mean(thetas_2d)
avg_phase = np.exp(1.0j*np.mean(thetas))
avg_v = np.exp(1.0j*2.0*np.arctan(np.mean(vs)))

# plot data
fig, ax = plt.subplots(figsize=(10, 10))
circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None')
ax.add_patch(circ)
x, y = coordinates(thetas_2d)
ax.scatter(x, y, marker="o", color="black")
kmo_x, kmo_y = coordinates(kmo)
ax.plot([0.0, kmo_x], [0.0, kmo_y], color="red", label="KO")
avg_x, avg_y = coordinates(avg_phase)
ax.plot([0.0, avg_x], [0.0, avg_y], color="green", label="mean(theta(v))")
v_x, v_y = coordinates(avg_v)
ax.plot([0.0, v_x], [0.0, v_y], color="royalblue", label="theta(mean(v))")
ax.legend()
ax.set_xlabel("real")
ax.set_ylabel("imag")
ax.set_title("Unit Circle")
plt.tight_layout()
plt.show()
