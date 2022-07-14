import numpy as np
from pandas import DataFrame
import pickle

# preparations
N = 7
conditions = np.arange(0, N)
D = np.zeros((N, N))
M = np.zeros((N, N), dtype=np.int32)
index = []
columns = []
in_vars = [0, 3, 6]
nodes = [300, 600]
ps = [3, 6]
dists = {}
for n in conditions:

    # load data
    res = pickle.load(open(f"results/rnn_simulations/rnn_{n}.p", "rb"))
    mod = pickle.load(open(f"results/rnn_modularity/mod_{n}.p", "rb"))

    # calculate variance around mean-field manifold
    for m in range(len(res['in_var'])):
        r = res['results'][m]['v']
        D[n, m] = np.var(r)
        M[n, m] = len(mod['modules'][m])
    index.append(res['p'])
    if n == 0:
        columns.extend(res['in_var'])

    # store input distributions
    if n in ps:
        dists[res['p']] = {}
        for v in in_vars:
            dists[res['p']][res['in_var'][v]] = {'dist': res['etas'][v],
                                                 's1': np.argwhere(res['W'][nodes[0], :] > 0).squeeze(),
                                                 's2': np.argwhere(res['W'][nodes[1], :] > 0).squeeze()}

# store results in Dataframe
D = DataFrame(data=D, index=index, columns=columns)
M = DataFrame(data=M, index=index, columns=columns)

# save results
pickle.dump({'D': D, 'M': M, 'dists': dists}, open("results/manifold_results.p", "wb"))
