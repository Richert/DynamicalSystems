from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pickle

# load data
data = pickle.load(open(f"results/eic/eic_rnn_0.p", "rb"))
print(fr"$\Delta = {data['delta_i']}$")

# peak detection
peaks, properties = find_peaks(-1.0*data['results']['ue'].squeeze(), width=50, prominence=0.4)
offset = 10000

# plotting
fig, ax = plt.subplots(nrows=2)
ax[0].plot(data['results']['ue'])
for i in range(len(peaks)):
    p = peaks[i]
    print(f"I = {data['I'][p+offset]}")
    ax[0].axvline(x=p, color='orange', linestyle='--')
ax[1].plot(data['I'])
plt.show()
