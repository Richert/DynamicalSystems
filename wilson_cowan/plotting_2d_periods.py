from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sb

df = read_csv("wc_2d_periods.csv")
df = df.drop(columns=["Unnamed: 0"])
fig, ax = plt.subplots(figsize=(8, 6))
sb.heatmap(df.pivot(values="period", index="S_i", columns="S_e"))
plt.show()
