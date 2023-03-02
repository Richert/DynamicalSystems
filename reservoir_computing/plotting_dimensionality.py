import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import pickle
import numpy as np
import os
import seaborn as sb
from pandas import DataFrame

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (12, 5.5)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6


# load data
###########

path = "results/dimensionality"
fn = "rs_dimensionality"
dimensionality = []
dim_columns = ["p", "Delta", "d", "dim", "modules"]

for file in os.listdir(path):
    if fn in file:
        f = pickle.load(open(f"{path}/{file}", "rb"))
        dim = f["dim"]
        mods = f["modules"]
        for i in range(dim.shape[0]):
            row = []
            dim_tmp = dim.iloc[i, :]
            row.append(f["sweep"]["p"])
            row.append(f["sweep"]["Delta"])
            row.append(dim_tmp["d"])
            row.append(dim_tmp["dim"])
            row.append(len(mods["m"][i]))
            dimensionality.append(row)

dimensionality = pd.DataFrame(columns=dim_columns, data=dimensionality)

# plotting
##########

# plot dimensionality bar graph for steady-state regime
g = sb.catplot(data=dimensionality, kind="bar", x="p", y="dim", hue="Delta", col="d",
               errorbar="sd", palette="dark", alpha=0.8,)
g.despine(left=True)
g.set_axis_labels("", "dimensionality")
g.legend.set_title("")

# # plot dimensionality bar graph for steady-state regime
# dim_lc = dimensionality.loc[dimensionality["d"] > 50.0, :]
# ax = fig.add_subplot(grid[0, 1])
# g = sb.catplot(data=dim_lc, kind="bar", x="p", y="dim", hue="Delta", errorbar="sd", palette="dark", alpha=0.8, ax=ax)
# g.despine(left=True)
# g.set_axis_labels("", "dimensionality")
# g.legend.set_title("")
# ax.set_title(r"Synchronous regime ($d = 10$)")

# finishing touches
###################

# padding
# fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)
#
# # saving/plotting
# fig.canvas.draw()
plt.savefig(f'results/dimensionality.pdf')
plt.show()
