import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

def gaussian(N, eta: float, Delta: float) -> np.ndarray:
    etas = eta + Delta * np.random.randn(N)
    return np.sort(etas)

def transform(x: np.ndarray, scale: float, noise: float):
    x_norm = np.abs(x) / np.max(np.abs(x))
    x_scaled = scale * x_norm**3
    degree = x_scaled + noise*np.random.randn(*x.shape)
    degree[degree < 0.0] = 0.0
    return degree

N = 200
noise = 3.0
scale = 10.0
thresholds = gaussian(N, -40.0, 5.0)
in_degree = transform(thresholds, scale, noise)

print(f"Plotting backend: {plt.rcParams['backend']}")
plt.rcParams["font.family"] = "sans"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (4, 2)
plt.rcParams['font.size'] = 28.0
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['lines.linewidth'] = 2.0
markersize = 2

sb.jointplot(x=thresholds, y=in_degree, kind="hex", color="#4CB391")
plt.xlabel("spike threshold")
plt.ylabel("in-degree")
plt.show()
