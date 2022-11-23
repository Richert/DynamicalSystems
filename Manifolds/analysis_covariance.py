import numpy as np
import matplotlib.pyplot as plt
import pickle


def get_dim(signals: list):
    dims = []
    for s in signals:
        s -= np.mean(s)
        s /= np.std(s)
        cov = s.T @ s
        cov[np.eye(cov.shape[0]) > 0] = 0.0
        eigs = np.abs(np.linalg.eigvals(cov))
        dims.append(np.sum(eigs)**2/np.sum(eigs**2))
    return np.mean(dims), np.std(dims)


# get filenames
path = 'results'
n_files = 7

ps, dim_hom, std_hom, dim_het, std_het, dim_med, std_med = [], [], [], [], [], [], []
for n in range(1, n_files):
    data_het = pickle.load(open(f"{path}/rnn_het_{n}.p", "rb"))
    data_med = pickle.load(open(f"{path}/rnn_med_{n}.p", "rb"))
    data_hom = pickle.load(open(f"{path}/rnn_hom_{n}.p", "rb"))
    ps.append(data_het['p'])
    mu, std = get_dim(data_hom["s"])
    data_hom["dim"] = (mu, std)
    dim_hom.append(mu)
    std_hom.append(std)
    mu, std = get_dim(data_het["s"])
    data_het["dim"] = (mu, std)
    dim_het.append(mu)
    std_het.append(std)
    mu, std = get_dim(data_med["s"])
    data_med["dim"] = (mu, std)
    dim_med.append(mu)
    std_med.append(std)
    pickle.dump(data_het, open(f"{path}/rnn_het_{n}.p", "wb"))
    pickle.dump(data_hom, open(f"{path}/rnn_hom_{n}.p", "wb"))
    pickle.dump(data_med, open(f"{path}/rnn_med_{n}.p", "wb"))

# number of communities
fig, ax = plt.subplots(figsize=(6, 4))
x1 = np.arange(-0.3, len(ps)-0.3, step=1)
x2 = np.arange(0.0, len(ps), step=1)
x3 = np.arange(0.3, len(ps)+0.3, step=1)
ax.bar(x1, dim_hom, yerr=std_hom, color='grey', width=0.2, label=r"$\Delta_v$ = 0.2")
ax.bar(x2, dim_med, yerr=std_het, color='blue', width=0.2, label=r"$\Delta_v$ = 1.0")
ax.bar(x3, dim_het, yerr=std_het, color='red', width=0.2, label=r"$\Delta_v$ = 2.0")
ax.set_xticks(np.arange(0, len(ps), step=2), labels=np.asarray(ps)[::2])
ax.set_xlabel(r'$p$')
ax.set_ylabel(r'$n$')
plt.show()
