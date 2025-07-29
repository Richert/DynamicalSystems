import numpy as np
import matplotlib.pyplot as plt


def ik_vf(v, C, k, v_r, v_t, eta):
    return (k*(v-v_r)*(v-v_t) + eta) / C


# parameters
C = 100.0
k = 0.7
v_r = -70.0
v_t = -40.0
eta = 0.0

thresholds = [-55.0, -40.0, -25.0]
etas = [-200.0, 0.0, 200.0]

# get phase portrait
v = np.linspace(-80.0, -20.0, num=1000)
eta_dvs = [ik_vf(v, C, k, v_r, v_t, eta_tmp) for eta_tmp in etas]
threshold_dvs = [ik_vf(v, C, k, v_r, threshold, eta) for threshold in thresholds]
combined_dvs = [[ik_vf(v, C, k, v_r, threshold, eta) for eta in etas] for threshold in thresholds]

# plotting
colors = ["darkorange", "darkblue", "darkgreen"]
linestyles = ["solid", "dashed", "dotted"]
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
ax = axes[0]
for eta, dv, c in zip(etas, eta_dvs, colors):
    ax.plot(v, dv, label=rf"$\eta = {eta}$ (pA)", color=c)
ax.axhline(xmin=0, xmax=1, y=0.0, color="black", linestyle="dashed")
ax.legend()
ax.set_xlabel(r"$v$ (mV)")
ax.set_ylabel(r"$\frac{dv}{dt}$ (pA)")
ax = axes[1]
for threshold, dv, c in zip(thresholds, threshold_dvs, colors):
    ax.plot(v, dv, label=rf"$v_t = {threshold}$ (mV)", color=c)
ax.axhline(xmin=0, xmax=1, y=0.0, color="black", linestyle="dashed")
ax.legend()
ax.set_xlabel(r"$v$ (mV)")
ax.set_ylabel(r"$\frac{dv}{dt}$ (pA)")
ax = axes[2]
for threshold, dvs, c in zip(thresholds, combined_dvs, colors):
    for eta, dv, style in zip(etas, dvs, linestyles):
        ax.plot(v, dv, label=rf"$v_t = {threshold}$ (mV), $\eta = {eta}$ (mV)", color=c, linestyle=style)
ax.axhline(xmin=0, xmax=1, y=0.0, color="black", linestyle="dashed")
# ax.legend()
ax.set_xlabel(r"$v$ (mV)")
ax.set_ylabel(r"$\frac{dv}{dt}$ (pA)")

plt.tight_layout()
plt.show()