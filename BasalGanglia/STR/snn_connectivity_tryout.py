from scipy.stats import rv_discrete, bernoulli
import numpy as np
import matplotlib.pyplot as plt


def wrap(idxs: np.ndarray, N: int) -> np.ndarray:
    idxs[idxs < 0] = N+idxs[idxs < 0]
    idxs[idxs >= N] = idxs[idxs >= N] - N
    return idxs


def generate_connectivity(N: int, p: float, spatial_distribution: rv_discrete) -> np.ndarray:
    C = np.zeros((N, N))
    n_conns = int(N*p)
    for n in range(N):
        idxs = spatial_distribution.rvs(size=n_conns)
        signs = 1 * (bernoulli.rvs(p=0.5, loc=0, size=n_conns) > 0)
        signs[signs == 0] = -1
        conns = wrap(n + idxs*signs, N)
        C[n, conns] = 1.0/n_conns
    return C


N = 1000
p = 0.1

xs = np.arange(0, N)
ys = 1/(xs**2+1)
ys /= np.sum(ys)
spatial_distribution = rv_discrete(b=N, name='spatial_kernel', values=(xs, ys))

C = generate_connectivity(N, p, spatial_distribution)
plt.imshow(C)

np.save('config/msn_conn.npy', C)
plt.show()
