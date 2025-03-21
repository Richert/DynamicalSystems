import numpy as np
import matplotlib.pyplot as plt

path = "/home/richard-gast/Documents/results/trujilo_2019"

# load data
proto_waves = np.load(f"{path}/trujilo_2019_cluster_centers_kmeans.npy")
D = np.load(f"{path}/trujilo_2019_waveform_distances.npy")

# plot prototypical waveforms
fig, ax = plt.subplots(figsize=(12, 5))
for sample in range(proto_waves.shape[0]):
    ax.plot(proto_waves[sample], label=sample)
ax.set_ylabel("firing rate")
ax.set_xlabel("time (ms)")
ax.legend()
ax.set_title("Prototypical waveforms fur clusters")
plt.tight_layout()

# plot distance matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(D, interpolation="none", aspect="equal", cmap="cividis")
plt.colorbar(im, ax=ax)
plt.show()
