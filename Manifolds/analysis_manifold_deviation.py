import numpy as np
from pandas import DataFrame
import pickle
import matplotlib.pyplot as plt

# calculate variance around mean-field manifold
N = 7
conditions = np.arange(0, N)
D = np.zeros((N, N))
index = []
columns = []
for n in conditions:
    res = pickle.load(open(f"results/rnn_simulations/rnn_{n}.p", "rb"))
    for m in range(len(res['in_var'])):
        r = res['results'][m]['v']
        D[n, m] = np.var(r)
    index.append(res['p'])
    if n == 0:
        columns.extend(res['in_var'])

# store results in Dataframe
results = DataFrame(data=D, index=index, columns=columns)

plt.imshow(results)
plt.colorbar()
plt.show()

# save results
results.to_pickle("results/manifold_deviations.p")
