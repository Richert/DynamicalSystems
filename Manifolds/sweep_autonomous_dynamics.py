import numpy as np
import pickle

# parameter sweep definition
sweep = dict()
sweep["eta"] = np.linspace(-100.0, 50.0, num=20)
sweep["g"] = np.linspace(0, 40, num=20)

# save sweep to file
fname = "sweep_dynamic_regime"
with open(f"config/{fname}.pkl", "wb") as f:
    pickle.dump(sweep, f)
    f.close()
