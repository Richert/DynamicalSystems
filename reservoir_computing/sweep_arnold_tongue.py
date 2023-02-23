import numpy as np
import pickle

# parameter sweep definition
sweep = dict()
sweep["alpha"] = np.linspace(0.0001, 0.002, num=20)
sweep["p_in"] = np.linspace(0.01, 0.99, num=20)

# save sweep to file
fname = "arnold_tongue_sweep"
with open(f"config/{fname}.pkl", "wb") as f:
    pickle.dump(sweep, f)
    f.close()
