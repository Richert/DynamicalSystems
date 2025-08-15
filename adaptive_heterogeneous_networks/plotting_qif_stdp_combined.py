import matplotlib.pyplot as plt
import pickle
import numpy as np

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 20.0
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# choose Deltas
deltas = [0.2, 0.8]
for delta in deltas:

    snn = pickle.load(open(f"/home/richard-gast/Documents/results/qif_snn_Delta_{int(delta*10.0)}.pkl", "rb"))
    fre = pickle.load(open(f"/home/richard-gast/Documents/results/qif_fre_Delta_{int(delta * 10.0)}.pkl", "rb"))

    # plotting
    fig = plt.figure()
    grid = fig.add_gridspec(nrows=3, ncols=2)

    # plotting snn connectivity
    W = fre["W"]
    ax = fig.add_subplot(grid[:2, 0])
    step = 8
    im = ax.imshow(W, aspect="auto", interpolation="none", cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("population")
    ax.set_ylabel("population")
    ax.set_xticks(ticks=np.arange(0, W.shape[0], step=step))
    ax.set_yticks(ticks=np.arange(0, W.shape[0], step=step))
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_title("FRE Connectivity")

    # plotting snn connectivity
    W = snn["W"]
    ax = fig.add_subplot(grid[:2, 1])
    step = 100
    im = ax.imshow(W, aspect="auto", interpolation="none", cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("neuron")
    ax.set_ylabel("neuron")
    ax.set_xticks(ticks=np.arange(0, W.shape[0], step=step))
    ax.set_yticks(ticks=np.arange(0, W.shape[0], step=step))
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_title("SNN Connectivity")

    # plotting average firing rate dynamics
    ax = fig.add_subplot(grid[2, :])
    fre_mean = np.mean(fre["res"]["s"].values, axis=1)
    snn_mean = np.mean(snn["s"], axis=1)
    ax.plot(fre["res"].index, fre_mean, label="FRE")
    ax.plot(fre["res"].index, snn_mean, label="SNN")
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("s")
    ax.set_title("Network dynamics")

    fig.suptitle(rf"$\Delta = {delta}$")
    fig.canvas.draw()
    plt.savefig(f"/home/richard-gast/Documents/results/qif_stdp_Delta_{int(delta*10)}.svg")
    plt.show()
