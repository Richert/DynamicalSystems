import pickle
import matplotlib.pyplot as plt
import numpy as np

# condition
cond = 0

# load data
fre = pickle.load(open("results/fre_results.p", "rb"))
snn = pickle.load(open(f"results/rnn_simulations/rnn_{cond}.p", "rb"))

# choose snn/fre simulations for specific eta
eta = 60
idx = np.argmin(np.abs(snn['etas'] - eta))
snn_res = np.mean(snn['results'][idx]['v'], axis=1)
idx = np.argmin(np.abs(fre['map'].loc['eta', :] - eta))
fre_res = fre['results'].loc[:, fre['map'].columns.values[idx]]

# calculate mean and variance of mean-field difference
diff = fre_res - snn_res
mean_diff = np.mean(diff)
std_diff = np.std(diff)
print(fr"Results for $\bar \eta = {eta}$:")
print(fr"--------------------------------")
print(fr"SNN and FRE signals show a mean difference in their average membrane potential of ${mean_diff}$ mV.")
print(fr"The fluctuations of the SNN signal around the average membrane potential of the FRE signal express a standard deviation of ${std_diff}$ mV.")

# plotting
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(fre_res.index, snn_res)
ax.plot(fre_res)
ax.set_ylabel('v (mV)')
ax.set_xlabel('time (ms)')
plt.legend(['SNN', 'FRE'])
plt.show()
