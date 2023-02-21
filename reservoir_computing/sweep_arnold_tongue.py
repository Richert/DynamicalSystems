import numpy as np
import pickle

# parameter sweep definition
sweep = dict()
sweep["alpha"] = np.linspace(0.0, 0.001, num=10)
sweep["p_in"] = np.linspace(0.01, 1.0, num=10)

# save sweep to file
fname = "arnold_tongue_sweep"
with open(f"config/{fname}.pkl", "wb") as f:
    pickle.dump(sweep, f)
    f.close()
