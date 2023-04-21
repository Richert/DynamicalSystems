import numpy as np
import pickle

# parameter sweep definition
sweep = dict()
sweep["Delta_i"] = np.asarray([0.2, 2.0])
sweep["trial"] = np.arange(0, 10)

# save sweep to file
fname = "bump_sweep"
with open(f"config/{fname}.pkl", "wb") as f:
    pickle.dump(sweep, f)
    f.close()
