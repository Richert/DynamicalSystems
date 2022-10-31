from rectipy import random_connectivity, normalize
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr
import pickle
from sklearn import linear_model as lm

plt.rc('text', usetex=True)


# load data
###########

data = pickle.load(open("results/snn_autonomous.pkl", "rb"))
etas = data["etas"]
J = data["J"]
CC = data["cc"]
N = len(etas)

# connectivity analysis
#######################

# calculate differences between background inputs
eta_diff = np.zeros_like(J)
for i in range(N):
    for j in range(i+1, N):
        eta_diff[i, j] = np.abs(etas[i] - etas[j])
        eta_diff[j, i] = eta_diff[i, j]

# calculate local fields
fields = [etas[np.argwhere(J[i, :] > 0).squeeze()] for i in range(N)]

# calculate differences in local field means
field_diff = np.zeros_like(J)
for i in range(N):
    for j in range(i+1, N):
        field_diff[i, j] = np.abs(np.mean(fields[i]) - np.mean(fields[j]))
        field_diff[j, i] = field_diff[i, j]

# calculate Kullback-Leibler divergence between local fields
eta_grid = np.linspace(np.min(etas), np.max(etas), num=100)
field_rhos = []
for i in range(N):
    grid_fields = np.zeros_like(eta_grid) + 1e-10
    for eta in fields[i]:
        idx = np.argmin(np.abs(eta_grid - eta))
        grid_fields[idx] += 1
    field_rhos.append(grid_fields/len(fields[i]))
field_kld = np.zeros_like(J)
for i in range(N):
    for j in range(i+1, N):
        kld1 = np.sum(rel_entr(field_rhos[i], field_rhos[j]))
        kld2 = np.sum(rel_entr(field_rhos[j], field_rhos[i]))
        field_kld[i, j] = np.min([kld1, kld2])
        field_kld[j, i] = field_kld[i, j]

# data normalization
####################

data_normalized = {}
predictors = ["J", "eta_diff", "field_diff", "field_kld"]
for key, mat in zip(predictors + ["cc"], [J, eta_diff, field_diff, field_kld, CC]):
    data_normalized[key] = normalize(mat, mode="minmax", row_wise=False)

# linear model
##############

y = data_normalized["cc"].flatten()
X = np.asarray([data_normalized[key].flatten() for key in predictors]).T
glm = lm.LinearRegression()
glm.fit(X, y)
results = {key: coef for key, coef in zip(predictors, glm.coef_)}
for key, val in results.items():
    print(f"Ridge regression coefficient for feature {key}: {val}")

Y_predict = np.zeros_like(data_normalized["cc"]) + glm.intercept_
for key in predictors:
    Y_predict += data_normalized[key] * results[key]

# plotting
##########

# plot data
for key, val in data_normalized.items():
    _, ax = plt.subplots()
    im = ax.imshow(val, aspect=1.0, interpolation="none")
    plt.colorbar(mappable=im)
    ax.set_title(key)

# predicted functional connectivity
_, ax = plt.subplots()
im = ax.imshow(Y_predict, aspect=1.0, interpolation="none")
plt.colorbar(mappable=im)
ax.set_title(r"$cc^*$")

plt.show()
