from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pickle

# load data
data = pickle.load(open(f"results/sfa/sfa_rnn_90.p", "rb"))
print(f"d = {data['d']}")

# peak detection
peaks, properties = find_peaks(-1.0*data['results']['u'].squeeze(), width=500, prominence=1.0)
offset = 10000

# plotting
fig, ax = plt.subplots(nrows=2)
ax[0].plot(data['results']['u'])
for i in range(len(peaks)):
    p = peaks[i]
    print(f"I = {data['I'][p+offset]}")
    ax[0].axvline(x=p, color='orange', linestyle='--')
ax[1].plot(data['I'])
plt.show()

