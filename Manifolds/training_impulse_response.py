import pandas as pd
from rectipy import readout
import numpy as np
import pickle
import sys
from scipy.ndimage import gaussian_filter1d

# load data
cond = int(sys.argv[-4])
p1 = str(sys.argv[-3])
p2 = str(sys.argv[-2])
path = str(sys.argv[-1])
fname = f"ir_{p1}_{p2}_{cond}"
data = pickle.load(open(f"{path}/{fname}.pkl", "rb"))
print(f"Condition: {data['sweep']}")


# create target data for different taus
#######################################

# create target data
sigma = 20
stimuli = np.asarray(data["stimuli"])
min_isi = np.min(np.abs(np.diff(stimuli)))
phis = np.linspace(0.1, 0.6, num=10)*min_isi
steps = int(data["T"]/(data["dt"]*data['sr']))
targets = []
for phi in phis:
    indices = stimuli + int(phi)
    if indices[-1] >= steps:
        indices = indices[:-1]
    I_tmp = np.zeros((steps,))
    I_tmp[indices] = 1.0
    targets.append(gaussian_filter1d(I_tmp, sigma=sigma))

# perform readout for each set of target data
#############################################

# create 2D dataframes
train_scores = pd.DataFrame(columns=phis, data=np.zeros((len(data["s"]), len(phis))))
test_scores = pd.DataFrame(columns=phis, data=np.zeros((len(data["s"]), len(phis))))
weights = []
intercepts = []
predictions_plotting = []
targets_plotting = []

# training procedure
cutoff = 1000
plot_length = 2000
for i, signal in enumerate(data["s"]):

    s = signal.iloc[cutoff:, :]

    weights_tmp = []
    intercepts_tmp = []
    predictions_plotting.append([])
    targets_plotting.append([])
    for j, (tau, target) in enumerate(zip(phis, targets)):

        # readout training
        res = readout(s, target[cutoff:], alpha=10.0, solver='lsqr', positive=False, tol=0.1, train_split=40000,
                      verbose=False)
        train_scores.iloc[i, j] = res['train_score']
        test_scores.iloc[i, j] = res['test_score']
        weights_tmp.append(res["readout_weights"])
        intercepts_tmp.append(res["readout_bias"])
        predictions_plotting[-1].append(res["prediction"])
        targets_plotting[-1].append(res["target"])

    weights.append(weights_tmp)
    intercepts.append(intercepts_tmp)

# save data to file
data["lags"] = phis
data["train_scores"] = train_scores
data["test_scores"] = test_scores
data["weights"] = weights
data["intercepts"] = intercepts
pickle.dump(data, open(f"{path}/{fname}.pkl", "wb"))
