import matplotlib.pyplot as plt
import pickle
import numpy as np

# load data
fn = "rnn_4"
data = pickle.load(open(f"results/rnn_simulations/{fn}.p", "rb"))

# plot signal
eta = 30
n = np.argmin(np.abs(data['etas']-eta))
state_var = "s"
indices = [50, 100, 200, 400, 800]
fig, ax = plt.subplots(figsize=(10, 6))
for idx in indices:
    ax.plot(data['results'][n][state_var][:, indices])
ax.set_xlabel('time')
ax.set_ylabel(state_var)
plt.legend(indices)
plt.show()
