import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sb
from pycobi import ODESystem

# plot settings
print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
markersize = 6

# file paths
path = "results"
auto_dir = "~/PycharmProjects/auto-07p"

# load pyauto data
a_rs = ODESystem.from_file(f"{path}/rs.pkl", auto_dir=auto_dir)
a_fs = ODESystem.from_file(f"{path}/fs.pkl", auto_dir=auto_dir)
a_lts = ODESystem.from_file(f"{path}/lts.pkl", auto_dir=auto_dir)

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = GridSpec(nrows=6, ncols=2, figure=fig)

