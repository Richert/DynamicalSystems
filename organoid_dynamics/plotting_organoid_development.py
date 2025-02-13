import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from pandas import read_csv

# load data
###########

dataset = "trujilo_2019"
path = f"/home/richard-gast/Documents/data/{dataset}"
file = f"{dataset}_summary.csv"
data = read_csv(f"{path}/{file}")

# plotting
##########

dvs = ["dim", "spike_reg", "burst_freq", "burst_reg", "rate_avg", "rate_het", "intraburst_dim",
       "intraburst_freq", "intraburst_spike_reg", "intraburst_rate_avg", "intraburst_rate_het",
       "lfp_dim", "lfp_var", "max_freq", "max_pow"]
for y in dvs:
    fig, ax= plt.subplots(figsize=(8, 4))
    sb.lineplot(data, x="age", y=y, hue="organoid", ax=ax)
plt.show()
