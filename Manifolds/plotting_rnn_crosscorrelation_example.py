import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.signal import correlate

# load data
fn = "rnn_3"
data = pickle.load(open(f"results/rnn_simulations/{fn}.p", "rb"))

# calculate crosscorrelation
eta = 70
n = np.argmin(np.abs(data['etas']-eta))
state_var = "s"
indices = [100, 600]
cc = correlate(data['results'][n][state_var][:, indices[0]], data['results'][n][state_var][:, indices[1]], method='fft',
               mode='same')

print(f"Maximum cross-correlation: {np.max(cc)}")

# plot signal
fig, axes = plt.subplots(nrows=2, figsize=(10, 7))
axes[0].plot(data['results'][n][state_var][:, indices])
axes[0].set_xlabel('time')
axes[0].set_ylabel(state_var)
plt.legend(indices)
axes[1].plot(cc)
axes[1].set_xlabel('time lag')
axes[1].set_ylabel('CC')
plt.show()
