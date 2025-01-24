import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from pandas import read_csv

# load data
###########

path = "/home/richard-gast/Documents/data/organoid_dynamics"
file = "organoid_dynamics_summary.csv"
data = read_csv(f"{path}/{file}")

# plotting
##########

dvs = ["dim", "isi_mean", "isi_std", "ibi_mean", "ibi_std", "fr_mean", "fr_std", "intraburst_dim",
       "intraburst_isi_mean", "intraburst_isi_std", "intraburst_fr_mean", "intraburst_fr_std"]
for y in dvs:
    fig, ax= plt.subplots(figsize=(8, 4))
    sb.lineplot(data, x="age", y=y, errorbar="sd", ax=ax)
plt.show()
