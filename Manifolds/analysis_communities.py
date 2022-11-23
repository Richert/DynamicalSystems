import numpy as np
import matplotlib.pyplot as plt
import pickle
from pyrecu import modularity


def z(x: np.ndarray):
    x -= np.mean(x)
    x /= np.std(x)
    return x


# get filenames
path = 'results'
n_files = 7

ps, dim_hom, std_hom, dim_het, std_het = [], [], [], [], []
for n in range(1, n_files):
    data_het = pickle.load(open(f"{path}/rnn_med_{n}.p", "rb"))
    data_hom = pickle.load(open(f"{path}/rnn_hom_{n}.p", "rb"))
    data_het["modules_new"], data_het["adjacency_new"], data_het["nodes_new"] = [], [], []
    data_hom["modules_new"], data_hom["adjacency_new"], data_hom["nodes_new"] = [], [], []
    ps.append(data_het['p'])
    signals_hom, signals_het = data_hom['s'], data_het['s']
    dim_hom_tmp, dim_het_tmp = [], []
    for s_hom, s_het in zip(signals_hom, signals_het):
        z_hom, z_het = z(s_hom), z(s_het)
        modules, adj, nodes = modularity(z_hom.T, threshold=0.1, min_connections=10, min_nodes=10,
                                         cross_corr_method='cov', decorator=None)
        data_hom["modules_new"].append(modules)
        data_hom["adjacency_new"].append(adj)
        data_hom["nodes_new"].append(nodes)
        dim_hom_tmp.append(len(modules))
        modules, _, _ = modularity(z_het.T, threshold=0.1, min_connections=10, min_nodes=10,
                                   cross_corr_method='cov', decorator=None)
        data_het["modules_new"].append(modules)
        data_het["adjacency_new"].append(adj)
        data_het["nodes_new"].append(nodes)
        pickle.dump(data_het, open(f"{path}/rnn_het_{n}.p", "wb"))
        pickle.dump(data_hom, open(f"{path}/rnn_hom_{n}.p", "wb"))
        dim_het_tmp.append(len(modules))

    dim_hom.append(np.mean(dim_hom_tmp))
    dim_het.append(np.mean(dim_het_tmp))
    std_hom.append(np.std(dim_hom_tmp))
    std_het.append(np.std(dim_het_tmp))

# number of communities
fig, ax = plt.subplots(figsize=(6, 4))
x1 = np.arange(-0.2, len(ps)-0.2, step=1)
x2 = np.arange(0.2, len(ps), step=1)
ax.bar(x1, dim_het, yerr=std_hom, color='grey', width=0.4)
ax.bar(x2, dim_hom, yerr=std_het, color='blue', width=0.4)
ax.set_xticks(np.arange(0, len(ps), step=2), labels=np.asarray(ps)[::2])
ax.set_xlabel(r'$p$')
ax.set_ylabel(r'$n$')
plt.show()
