from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pickle
import numpy as np

# load data
data = pickle.load(open(f"results/eic2/eic_rnn2_90.p", "rb"))
print(fr"$\Delta = {data['delta_i']}$")

# filter data
filtered = gaussian_filter1d(data['results']['ui'].squeeze(), sigma=50)

# peak detection
hb_peaks, _ = find_peaks(filtered, width=100, prominence=0.03)
lp_peaks, properties = find_peaks(-1.0*filtered, width=1000, prominence=0.2)
offset = 10000

# plotting
fig, ax = plt.subplots(nrows=2)
ax[0].plot(filtered)
for i in range(len(hb_peaks)):
    p = hb_peaks[i]
    print(f"I_hb{i} = {data['I'][p+offset]}")
    ax[0].axvline(x=p, color='orange', linestyle='--')
try:
    print(f"I_lp1 = {data['I'][lp_peaks[0] - int(properties['widths'][0]) + offset]}")
    print(f"I_lp2 = {data['I'][lp_peaks[-1] + offset]}")
    ax[0].axvline(x=lp_peaks[0] - 0.5*int(properties['widths'][0]), color='green', linestyle='--')
    ax[0].axvline(x=lp_peaks[-1], color='green', linestyle='--')
except IndexError:
    pass
ax[1].plot(data['I'])
plt.show()
