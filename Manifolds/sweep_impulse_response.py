import numpy as np
import pickle

# parameter sweep definition
sweep = dict()
sweep["alpha"] = np.asarray([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0])
sweep["Delta"] = np.asarray([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
sweep["p"] = np.asarray([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64])

# save sweep to file
fname = "impulse_response_sweep"
with open(f"config/{fname}.pkl", "wb") as f:
    pickle.dump(sweep, f)
    f.close()
