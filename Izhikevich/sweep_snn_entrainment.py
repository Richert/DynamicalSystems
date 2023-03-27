import numpy as np
import pickle

# parameter sweep definition
sweep = dict()
sweep["Delta"] = np.linspace(0.01, 2.0, num=20)
sweep["trial"] = np.arange(0, 10)

# save sweep to file
fname = "entrainment_sweep"
with open(f"config/{fname}.pkl", "wb") as f:
    pickle.dump(sweep, f)
    f.close()
