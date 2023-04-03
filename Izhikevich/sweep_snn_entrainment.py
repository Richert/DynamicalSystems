import numpy as np
import pickle

# parameter sweep definition
sweep = dict()
sweep["Delta"] = np.linspace(0.01, 2.0, num=20)
sweep["trial"] = np.arange(0, 10)

# load data that relate Delta to omega
coh = pickle.load(open(f"results/mf_entrainment_lc.pkl", "rb"))["coherence"]
threshold = 0.9
omegas = []
for delta in sweep["Delta"]:
    row = np.argmin(np.abs(coh.index - delta)).squeeze()
    col = coh.shape[1]-1
    while coh.iloc[row, col] < threshold:
        col -= 1
    omegas.append(coh.columns.values[col])
sweep["omega"] = omegas

# save sweep to file
fname = "entrainment_sweep"
with open(f"config/{fname}.pkl", "wb") as f:
    pickle.dump(sweep, f)
    f.close()
