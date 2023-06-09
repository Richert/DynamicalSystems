import numpy as np
import matplotlib.pyplot as plt
import h5py


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.squeeze() / np.max(x.flatten())
    y = y.squeeze() / np.max(y.flatten())
    return float(np.mean((x - y)**2))


# load data
###########

cond = "9"
data_set = "4"
path = "results/funcgen"

# load examples
examples = {}
f = f"snn_funcgen_{cond}.h5"
data = h5py.File(f"{path}/{f}", "r")
g = data["sweep"]
delta = np.round(np.asarray(g["Delta"]), decimals=2)
g = data[data_set]
alpha = np.round(np.asarray(g["alpha"]), decimals=1)
# examples["s"] = np.asarray(g["s"])
examples["target"] = np.asarray(g["targets"])
examples["test_prediction"] = np.asarray(g["test_predictions"])
examples["train_prediction"] = np.asarray(g["train_predictions"])
# examples["train_phases"] = np.asarray(g["train_phases"])
# examples["test_phases"] = np.asarray(g["test_phases"])
examples["dt"] = np.asarray(g["dt"])
examples["sr"] = np.asarray(g["sr"])
examples["input_indices"] = np.asarray(g["input_indices"])
examples["dim"] = np.round(np.asarray(g["dimensionality"]), decimals=1)
# examples["K"] = np.asarray(g["K"])
examples["K_mean"] = np.asarray(g["K_mean"])
examples["K_var"] = np.asarray(g["K_var"])
examples["K_diag"] = np.asarray(g["K_diag"])

############
# plotting #
############

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6
ticks = 5

# create figure layout
fig = plt.figure(figsize=(10, 8), constrained_layout=True)
grid = fig.add_gridspec(4, 2)

# SNN dynamics
# width = int(20.0/(examples["dt"]*examples["sr"]))
# indices = examples["input_indices"]
# ax = fig.add_subplot(grid[0, 0])
# s = examples["s"][np.arange(0, len(examples["s"]), 4)]
# s_all = np.concatenate(s, axis=1)
# s_all /= np.max(s_all)
# phases = np.round(np.mod(np.arange(0, s_all.shape[1]), s[0].shape[1])*np.pi*2.0/s[0].shape[1], decimals=1)
# phase_ticks = np.arange(0, len(phases), 1111)
# im = ax.imshow(s_all, aspect="auto", interpolation="none", cmap="Greys")
# plt.sca(ax)
# dur = 0
# for n in range(len(s)):
#     plt.fill_betweenx(y=indices, x1=[dur for _ in range(len(indices))],
#                       x2=[width + dur for _ in range(len(indices))], color='red', alpha=0.5)
#     dur += len(s[n, 0])
#     ax.axvline(x=dur, color="blue", linestyle="solid")
# ax.set_xticks(phase_ticks, labels=phases[phase_ticks])
# ax.set_xlabel('phase')
# ax.set_ylabel('neurons')
# ax.set_title(fr"(A) Network dynamics for {len(s)} trials")

# K_mean
ax = fig.add_subplot(grid[0, 0])
ax.plot(examples["K_mean"])
ax.set_title("K_mean")

# K_var
ax = fig.add_subplot(grid[0, 1])
ax.plot(examples["K_var"])
ax.set_title("K_var")

# predictions
trials = [0, 2, 4]

for i, ex in enumerate(trials):

    # target 1
    ax = fig.add_subplot(grid[i+1, 0])
    ax.plot(examples["target"][1], label="target", color="black")
    ax.plot(examples["train_prediction"][1], label="fit", color="blue")
    ax.plot(examples["test_prediction"][1][ex], label="prediction", color="orange")
    if i == 0:
        ax.set_title(f"(B) Function generation performance on target 1")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # target 2
    ax = fig.add_subplot(grid[i+1, 1])
    target = examples["target"][0]
    tmax = np.max(target)
    ax.plot(target / tmax, label="target", color="black")
    ax.plot(examples["train_prediction"][0] / tmax, label="fit", color="blue")
    ax.plot(examples["test_prediction"][0][ex] / tmax, label="prediction", color="orange")
    ax.set_xlabel("time")
    ax.set_ylabel("")
    if i == 0:
        ax.set_title(f"(C) Function generation performance on target 2")
    ax.legend()

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
plt.suptitle(f"alpha = {alpha}, delta = {delta}")
# saving/plotting
fig.canvas.draw()
plt.show()
