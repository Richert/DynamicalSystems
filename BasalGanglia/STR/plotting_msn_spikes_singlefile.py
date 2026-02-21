import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
import os

def get_eigs(rates: np.ndarray, epsilon: float = 1e-12) -> tuple:

    rates_centered = np.zeros_like(rates)
    for i in range(rates.shape[1]):
        rates_centered[:, i] = rates[:, i] - np.mean(rates[:, i])
        rates_centered[:, i] /= (np.std(rates[:, i]) + epsilon)
    C = np.cov(rates_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)
    return C, eigvals[idx], eigvecs[:, idx]

def get_ff(rates: np.ndarray) -> np.ndarray:
    n = rates.shape[1]
    ff = np.zeros((n,))
    for i in range(n):
        ff[i] = np.var(rates[:, i]) / np.mean(rates[:, i])
    return ff

# choose condition to plot
drug = "clozapine"
dose = "Vehicle"
mouse_id = "971"
fields = ["events_5hz", "dff_traces_5hz", "speed_traces_5hz", "cellXYcoords_um"]

# load data
data = {}
path = f"/mnt/kennedy_lab_data/Parkerlab/{drug}_neuraldata"
for file in os.listdir(f"{path}/{drug}/{dose}"):
    if mouse_id in file:
        condition = "amph" if "amph" in file else "veh"
        data_tmp = loadmat(f"{path}/{drug}/{dose}/{file}/{condition}_drug.mat", simplify_cells=True)
        data[condition] = {field: data_tmp[f"{condition}_drug"][field] for field in fields}

# calculate neuron covariances and spatial distances
for key in data:

    # covariance
    traces = data[key][fields[1]]
    C, eigvals, eigvecs = get_eigs(traces.T)
    eig_traces = eigvecs.T @ traces
    data[key]["cov"] = C
    data[key]["pc1"] = eig_traces[0, :]
    data[key]["pc2"] = eig_traces[1, :]

    # distance
    coords = data[key][fields[-1]]
    mean_coords = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - mean_coords, axis=1)
    sorting_idx = np.argsort(distances)
    data[key]["sorting_idx"] = sorting_idx
    data[key]["dist"] = squareform(pdist(coords, metric="euclidean"))

# plotting
n_ticks = 5
sr = 5
max_neurons = 200
for key in data:

    spikes = data[key][fields[0]]
    traces = data[key][fields[1]]
    speed = data[key][fields[2]]
    n_steps = spikes.shape[1]
    xticks = np.linspace(0, n_steps, n_ticks)

    # calculate

    # plot network dynamics vs. movement speed
    fig, axes = plt.subplots(nrows=3, figsize=(12, 6), sharex=True)
    ax = axes[0]
    ax.imshow(spikes[:max_neurons, :], aspect="auto", cmap="Greys", vmin=0.0, vmax=1.0, interpolation="none")
    ax.set_xticks(xticks, labels=[int(tick/sr) for tick in xticks])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neurons")
    ax.set_title(f"Spikes for condition: {key}")
    ax = axes[1]
    ax.plot(np.mean(traces, axis=0)*sr, label="mean(traces)")
    ax.plot(data[key]["pc1"]*sr, label="pc1")
    ax.plot(data[key]["pc2"]*sr, label="pc2")
    ax.set_xticks(xticks, labels=[int(tick / sr) for tick in xticks])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dF/F")
    ax.legend()
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
    C = data[key]["cov"][:max_neurons, :max_neurons]
    idx = data[key]["sorting_idx"]
    C = C[idx, :]
    C = C[:, idx]
    im = ax.imshow(C, aspect="equal", cmap="viridis", interpolation="none")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("neuron")
    ax.set_ylabel("neuron")
    ax.set_title(f"Covariance for condition: {key}")
    ax = axes[1]
    D = data[key]["dist"][:max_neurons, :max_neurons]
    D = D[idx, :]
    D = D[:, idx]
    im = ax.imshow(D, aspect="equal", cmap="viridis", interpolation="none")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("neuron")
    ax.set_ylabel("neuron")
    ax.set_title(f"Euclidean distance for condition: {key}")
    fig.suptitle(f"Mouse: {mouse_id}, drug: {drug}, dose: {dose}")
    plt.tight_layout()

plt.show()
