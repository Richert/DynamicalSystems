import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
import os

# choose condition to plot
drug = "SKF38393"
dose = "Vehicle"
mouse_id = "971"
fields = ["events_5hz", "dff_traces_5hz", "speed_traces_5hz", "cellXYcoords_um"]

# load data
data = {}
path = "/run/user/1000/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Basic_Sciences/Phys/Kennedylab/Parkerlab/Calcium_v2"
for file in os.listdir(f"{path}/{drug}/{dose}"):
    if mouse_id in file:
        condition = "amph" if "amph" in file else "veh"
        data_tmp = loadmat(f"{path}/{drug}/{dose}/{file}/{condition}_drug.mat", simplify_cells=True)
        data[condition] = {field: data_tmp[f"{condition}_drug"][field] for field in fields}

# calculate neuron covariances and spatial distances
for key in data:

    # covariance
    traces = data[key][fields[0]]
    data[key]["cov"] = traces @ traces.T

    # distance
    coords = data[key][fields[-1]]
    data[key]["dist"] = squareform(pdist(coords, metric="euclidean"))

# plotting
n_ticks = 5
sr = 5
max_neurons = 200
for key in data:

    spikes = data[key][fields[0]]
    speed = data[key][fields[2]]
    n_steps = spikes.shape[1]
    xticks = np.linspace(0, n_steps, n_ticks)

    # plot network dynamics vs. movement speed
    fig, axes = plt.subplots(nrows=3, figsize=(12, 6), sharex=True)
    ax = axes[0]
    ax.imshow(spikes[:max_neurons, :], aspect="auto", cmap="Greys", vmin=0.0, vmax=0.2, interpolation="gaussian")
    ax.set_xticks(xticks, labels=[int(tick/sr) for tick in xticks])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neurons")
    ax.set_title(f"Spikes for condition: {key}")
    ax = axes[1]
    ax.plot(np.mean(spikes, axis=0)*sr)
    ax.set_xticks(xticks, labels=[int(tick / sr) for tick in xticks])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("spike rate (Hz)")
    ax.set_title(f"Population rate for condition: {key}")
    ax = axes[2]
    ax.plot(speed)
    ax.set_xticks(xticks, labels=[int(tick / sr) for tick in xticks])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed")
    ax.set_title(f"Mouse running velocity for condition: {key}")
    fig.suptitle(f"Mouse: {mouse_id}, drug: {drug}, dose: {dose}")
    plt.tight_layout()

    # plot neuron covariance vs spatial distance
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    ax = axes[0]
    im = ax.imshow(data[key]["cov"][:max_neurons, :max_neurons], aspect="equal", cmap="viridis", interpolation="none")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("neuron")
    ax.set_ylabel("neuron")
    ax.set_title(f"Covariance for condition: {key}")
    ax = axes[1]
    im = ax.imshow(data[key]["dist"][:max_neurons, :max_neurons], aspect="equal", cmap="viridis", interpolation="none")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("neuron")
    ax.set_ylabel("neuron")
    ax.set_title(f"Euclidean distance for condition: {key}")
    fig.suptitle(f"Mouse: {mouse_id}, drug: {drug}, dose: {dose}")
    plt.tight_layout()

plt.show()
