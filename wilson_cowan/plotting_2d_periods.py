import numpy as np
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import seaborn as sb

# read data
df = read_csv("wc_2d_periods.csv")
df = df.drop(columns=["Unnamed: 0"])

# transform data into 2D format
s_e_unique = np.unique(df["S_e"])
s_i_unique = np.unique(df["S_i"])
df_2d = DataFrame(columns=s_e_unique, index=s_i_unique, data=np.zeros((len(s_i_unique), len(s_e_unique))))
for idx in df.index:
    s_e = df.at[idx, "S_e"]
    s_i = df.at[idx, "S_i"]
    period = df.at[idx, "period"]
    idx_r = np.argmin(np.abs(s_i_unique - s_i))
    idx_c = np.argmin(np.abs(s_e_unique - s_e))
    df_2d.iloc[idx_r, idx_c] = period

fig, ax = plt.subplots(figsize=(8, 6))
sb.heatmap(df_2d)
ax.set_xlabel("S_e")
ax.set_ylabel("S_i")
plt.show()
