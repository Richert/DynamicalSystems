import sys
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
sys.path.append('../')
import pickle

# load data
data = pickle.load(open(f"results/sfa_rnn_high.p", "rb"))['results']

# peak detection
peaks, properties = find_peaks(-1.0*data['u'].squeeze(), width=500, prominence=2.0)

# plotting
fig, ax = plt.subplots()
ax.plot(data['u'])
for p in peaks:
    ax.axvline(x=p, color='orange', linestyle='--')
plt.show()
