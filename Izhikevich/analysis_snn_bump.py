import pickle
import numpy as np
from scipy.stats import wasserstein_distance
import pandas as pd
import os


def outside_bump_activity(x: np.ndarray, y: np.ndarray) -> float:
    target_mean = np.mean(y)
    idx = np.argwhere(y < 0.5*target_mean).squeeze()
    return np.mean(x[idx]).squeeze() if len(idx) > 0 else 0.0


def total_bump_difference(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return np.sqrt(np.mean(np.inner(diff, diff)).squeeze())


# file path and name
path = "results/bump_new"
fn = "snn_bump"

# parameters of interest
p1 = "Delta"
p2 = "p_in"

# calculate KLDs
for file in os.listdir(path):
    if fn in file:

        results = {"within_bump_dist": [], "outside_bump_dist": [], p1: [], p2: []}

        # extract data
        data = pickle.load(open(f"{path}/{file}", "rb"))
        targets = data["target_dists"]
        snn_data = data["population_dists"]
        v1 = data["sweep"][p1]
        v2s = data[p2]

        # calculate KLD
        for x, y, v2 in zip(snn_data, targets, v2s):
            x = x / np.max(x)
            y = y / np.max(y)
            results["within_bump_dist"].append(total_bump_difference(x, y))
            results["outside_bump_dist"].append(outside_bump_activity(x, y))
            results[p1].append(v1)
            results[p2].append(v2)

        # store results
        data["bumps"] = results
        pickle.dump(data, open(f"{path}/{file}", "wb"))
