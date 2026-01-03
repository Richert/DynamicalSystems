import pickle

import numpy as np
import torch
from sbi import utils as utils
from sbi.inference import NPE
import os

# parameter definitions
#######################

# define directory and model
path = "/home/richard/data/sbi_organoids"
model = "ik_full"

# sbi parameters
device = "cpu"
estimator = "mdn"
stop_after_epochs = 30
clip_max_norm = 10.0
lr = 5e-5

# add simulated data to inference object
########################################

# create prior
params = pickle.load(open(f"{path}/{model}_parameters.pkl", "rb"))
param_bounds = params["bounds"]
param_keys = params["parameters"]
prior_min = [param_bounds[key][0] for key in param_keys]
prior_max = [param_bounds[key][1] for key in param_keys]
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max), device=device)

# create inference object
inference = NPE(prior=prior, density_estimator=estimator, device=device)

# add simulations
results = {"x": [], "theta": []}
for file in os.listdir(f"{path}"):
    if file.endswith(".pkl") and f"{model}_results" in file:
        try:
            data = pickle.load(open(f"{path}/{file}", "rb"))
            results["theta"].append(data["theta"])
            results["x"].append(data["x"])
        except (EOFError, pickle.UnpicklingError):
            pass
inference = inference.append_simulations(
    torch.tensor(np.asarray(results["theta"]), device=device, dtype=torch.float32),
    torch.tensor(np.asarray(results["x"]), device=device, dtype=torch.float32)
)

# fitting procedure
###################

density_estimator = inference.train(stop_after_epochs=stop_after_epochs, clip_max_norm=clip_max_norm,
                                    learning_rate=lr)
posterior = inference.build_posterior(density_estimator)
pickle.dump(posterior, open(f"{path}/{model}_posterior.pkl", "wb"))
