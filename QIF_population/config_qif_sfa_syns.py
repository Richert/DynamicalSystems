import numpy as np
import pickle
from rectipy import random_connectivity

fname = 'data/qif_sfa_syns_config.pkl'

# simulation parameters
#######################

T = 1125.0
dt = 1e-3
dts = 1e-1
cutoff = 125.0
start = 125.0

# network configuration parameters
##################################

N = 1000
p = 0.1
m = 5

# setup connectivity matrix
C = random_connectivity(N, N, p, normalize=True)

# setup input matrix
p_in = 0.05
W_in = random_connectivity(N, m, p_in, normalize=False)
for col in range(m):
    w_sum = np.sum(W_in[:, col])
    W_in[:, col] -= w_sum
print(np.sum(W_in, axis=0))

# define network input
######################

n_epochs = 10
input_rate = 30.0/100.0

steps = int(np.round(T/dt))
store_steps = int(np.round((T - cutoff)/dts))

inp_start = int(np.round(start/dt))
epoch_steps = int(np.floor((steps-inp_start)/n_epochs))
input_dur = int(np.floor(epoch_steps/m))

store_start = 0
store_epoch = int(np.floor((store_steps-store_start)/n_epochs))
store_dur = int(np.floor(store_epoch/m))

inp = np.zeros((m, steps))
targets = np.zeros((m, store_steps))

for i in range(n_epochs):
    for j in range(m):

        # generate spike train
        spike_train = np.random.poisson(input_rate, (input_dur,))

        # add spike train to input array
        idx = np.arange(inp_start+(i*m+j)*input_dur, inp_start+(i*m+j+1)*input_dur)
        inp[j, idx] = spike_train

        # store target output
        idx = np.arange(store_start+(i*m+j)*store_dur, store_start+(i*m+j+1)*store_dur)
        targets[j, idx] = 1.0

import matplotlib.pyplot as plt
plt.imshow(inp, aspect='auto', interpolation='none')
plt.colorbar()
plt.show()

# store data
data = {}
data['T'] = T
data['dt'] = dt
data['dts'] = dts
data['cutoff'] = cutoff
data['N'] = N
data['p'] = p
data['C'] = C
data['W_in'] = W_in
data['inp'] = inp
data['targets'] = targets.T
data['number_input_channels'] = m
pickle.dump(data, open(fname, 'wb'))
