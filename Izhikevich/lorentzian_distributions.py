import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt

# get Lorentzian random variables
mus = [-0.1, 0.1]
deltas = [0.2, 0.05]
bounds = (-2.0, 2.0)
x = np.linspace(bounds[0], bounds[1], num=10000)
pdfs = [cauchy.pdf(x, loc=mu, scale=delta) for mu, delta in zip(mus, deltas)]

# get PDFs
bins = 1000
edges = np.linspace(bounds[0], bounds[1], num=bins)
distributions = [np.histogram(pdf, bins=edges)[0] for pdf in pdfs]
distributions = [d / np.sum(d) for d in distributions]

# plotting
fig, ax = plt.subplots(figsize=(12, 4))
width = (np.abs(bounds[0]) + np.abs(bounds[1]))/ (bins * 1.1)
ax.bar(edges[1:], distributions[0], color="blue", label="L1", alpha=0.5, width=width)
ax.bar(edges[1:], distributions[1], color="red", label="L2", alpha=0.5, width=width)
combined = (distributions[0] + distributions[1])
combined /= np.sum(combined)
# ax.bar(edges[1:], combined, color="green", label="L1 + L2", alpha=0.5, width=width)
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("p")
plt.tight_layout()
plt.show()
