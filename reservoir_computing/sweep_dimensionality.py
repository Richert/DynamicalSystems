import numpy as np
import pickle

# parameter sweep definition
sweep = dict()
sweep["Delta"] = [0.1, 0.5, 1.0]
sweep["p"] = 1/2**np.arange(6)

# save sweep to file
fname = "dimensionality_sweep"
with open(f"config/{fname}.pkl", "wb") as f:
    pickle.dump(sweep, f)
    f.close()
