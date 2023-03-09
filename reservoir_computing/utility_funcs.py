import numpy as np
from scipy.signal import butter, sosfilt, hilbert
from scipy.stats import cauchy


def butter_bandpass_filter(data: np.ndarray, freqs: tuple, fs: int, order: int) -> np.ndarray:
    sos = butter(order, freqs, btype="bandpass", output="sos", fs=fs)
    return sosfilt(sos, data)


def analytic_signal(sig: np.ndarray) -> tuple:
    sig_analytic = hilbert(sig)
    sig_phase = np.unwrap(np.angle(sig_analytic))
    sig_envelope = np.abs(sig_analytic)
    return sig_phase, sig_envelope


def phase_locking(x: np.ndarray, y: np.ndarray) -> float:
    return np.abs(np.mean(np.exp(1.0j*(x-y))))


def coherence(x_phase: np.ndarray, y_phase: np.ndarray, x_env: np.ndarray, y_env: np.ndarray) -> float:
    coh = np.abs(np.sum(x_env * y_env * np.exp(1.0j*(x_phase - y_phase))))
    x_env_denom = np.sum(x_env**2)
    y_env_denom = np.sum(y_env**2)
    denom = np.sqrt(x_env_denom*y_env_denom)
    if denom > 0:
        return coh / denom
    else:
        return 0.0


def get_dim(s: np.ndarray):
    s = s - np.mean(s)
    s = s / np.std(s)
    cov = s.T @ s
    cov[np.eye(cov.shape[0]) > 0] = 0.0
    eigs = np.abs(np.linalg.eigvals(cov))
    return np.sum(eigs)**2/np.sum(eigs**2), cov


def lorentzian(n: int, eta: float, delta: float, lb: float, ub: float):
    samples = np.zeros((n,))
    for i in range(n):
        s = cauchy.rvs(loc=eta, scale=delta)
        while s <= lb or s >= ub:
            s = cauchy.rvs(loc=eta, scale=delta)
        samples[i] = s
    return samples


def dist(x: int, method: str = "inverse") -> float:
    if method == "inverse":
        return 1/x if x > 0 else 1
    if method == "inverse_squared":
        return 1/x**2 if x > 0 else 1
    if method == "exp":
        return np.exp(-x)
    else:
        raise ValueError("Invalid method.")


def fit_lorentzian(x: np.ndarray, modules: dict, nodes: list = None) -> tuple:
    deltas = []
    mus = []
    if nodes:
        x = x[nodes]
    for indices in modules.values():
        x_mod = x[indices]
        mu, delta = cauchy.fit(x_mod)
        deltas.append(delta)
        mus.append(mu)
    return np.asarray(deltas), np.asarray(mus)


def get_module_coupling(W: np.ndarray, modules: dict, nodes: list = None) -> np.ndarray:
    if nodes:
        W = W[nodes, :]
        W = W[:, nodes]
    W_mod = np.zeros((len(modules), len(modules)))
    for i, mod1 in enumerate(modules):
        targets = modules[mod1]
        for j, mod2 in enumerate(modules):
            sources = modules[mod2]
            W_tmp = W[targets, :]
            W_tmp = W_tmp[:, sources]
            W_mod[i, j] = np.mean(np.sum(W_tmp, axis=1))
    W_mod /= np.sum(W_mod, axis=1, keepdims=False)
    return W_mod
