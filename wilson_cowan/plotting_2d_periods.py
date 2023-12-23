import numpy as np
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import seaborn as sb
from pycobi import ODESystem


def interpolate2d(x, val=0.0):
    m, n = x.shape
    for i in range(m):
        for j in range(n):
            if i != 0 and j != 0 and i != m-1 and j != n-1:
                vals = np.asarray([x[i-1, j-1], x[i, j-1], x[i+1, j-1], x[i-1, j], x[i+1, j], x[i-1, j+1], x[i, j+1],
                                   x[i+1, j+1]])
                if np.sum(vals == val) < 3 and x[i, j] < np.mean(vals):
                    x[i, j] = np.mean(vals[vals != val])
    return x


# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['lines.linewidth'] = 1.5
markersize = 10

# read data
df = read_csv("wc_2d_periods.csv")
df = df.drop(columns=["Unnamed: 0"])
ode = ODESystem.from_file("wc_bifurcations.pkl", auto_dir="~/PycharmProjects/auto-07p")

# transform data into 2D format
s_e_unique = np.unique(df["S_e"])
s_i_unique = np.unique(df["S_i"])
data = np.zeros((len(s_i_unique), len(s_e_unique)))
df_2d = DataFrame(columns=s_e_unique, index=s_i_unique, data=data)
for idx in df.index:
    s_e = df.at[idx, "S_e"]
    s_i = df.at[idx, "S_i"]
    period = df.at[idx, "period"]
    idx_r = np.argmin(np.abs(s_i_unique - s_i))
    idx_c = np.argmin(np.abs(s_e_unique - s_e))
    data[idx_r, idx_c] = period
data = interpolate2d(data)
data[data < 1e-6] = np.nan
df_2d = DataFrame(columns=s_e_unique, index=s_i_unique, data=data)
fig, axes = plt.subplots(nrows=2, figsize=(8, 10))

ax = axes[0]
ode.plot_continuation("E/wc_e/s", "I/wc_i/s", cont="s_e/s_i:hb1", ax=ax, color="red", ignore=["UZ"],
                      line_style_unstable="solid", line_color_stable='#148F77', line_color_unstable='#148F77')
ode.plot_continuation("E/wc_e/s", "I/wc_i/s", cont="s_e/s_i:hb2", ax=ax, color="red", ignore=["UZ"],
                      line_style_unstable="solid", line_color_stable='#148F77', line_color_unstable='#148F77')
ode.plot_continuation("E/wc_e/s", "I/wc_i/s", cont="s_e/s_i:lp1", ax=ax, color="blue", ignore=["UZ"],
                      line_style_unstable="solid", line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E')
ode.plot_continuation("E/wc_e/s", "I/wc_i/s", cont="s_e/s_i:lp2", ax=ax, color="blue", ignore=["UZ"],
                      line_style_unstable="solid", line_color_stable='#5D6D7E', line_color_unstable='#5D6D7E')
ax.axvline(np.min(s_e_unique), ymin=np.min(s_i_unique), ymax=np.max(s_i_unique))
ax.axvline(np.max(s_e_unique), ymin=np.min(s_i_unique), ymax=np.max(s_i_unique))
ax.axhline(np.min(s_i_unique), xmin=-5.0, xmax=10.0)
ax.axhline(np.max(s_i_unique), xmin=-5.0, xmax=10.0)
ax.set_title(r"$a = 16$, $e = 15$")
ax.set_xlabel(r"$S_E$")
ax.set_ylabel(r"$S_I$")
ax.set_ylim([np.min(s_i_unique) - 0.5, np.max(s_i_unique) + 0.5])

ax = axes[1]
sb.heatmap(df_2d.iloc[::-1, :], ax=ax, rasterized=True, linewidths=0.0)
ax.set_xlabel(r"$S_E$")
ax.set_ylabel(r"$S_I$")

plt.tight_layout()
fig.canvas.draw()
plt.savefig("wc_bifurcations.svg")
plt.show()
