import pickle
import numpy as np
import sys

# load data
fn = sys.argv[-1] #"results/rs_arnold_tongue_hom.pkl"
data = pickle.load(open(fn, "rb"))

# input definition
cutoff = 20000.0
T = 200000.0 + cutoff
dt = 1e-2
sr = 100
steps = int(np.round(T/dt))
time = np.linspace(0.0, T, num=steps)
omega = 0.005
I_ext = np.sin(2.0*np.pi*omega*time)

# check whether crucial info exists in results file. if not, add it to the file.
check_keys = ["omega", "p", "Delta", "I_ext", "sr"]
check_vals = [omega, 0.1, 1.0, I_ext[::sr], sr]
for key, val in zip(check_keys, check_vals):
    if key not in data:
        data[key] = val

# save data
with open(fn, "wb") as f:
    pickle.dump(data, f)
    f.close()
