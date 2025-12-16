import pickle
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
device = "cuda:0"
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
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

# create inference object
inference = NPE(prior=prior, density_estimator=estimator, device=device)

# add simulations
for file in os.listdir(f"{path}"):
    if file.endswith(".pkl") and f"{model}_results" in file:
        data = pickle.load(open(f"{path}/{file}", "rb"))
        theta = torch.tensor(data["theta"], device=device)
        x = torch.tensor(data["x"], device=device)
        inference = inference.append_simulations(theta, x)

# fitting procedure
###################

density_estimator = inference.train(stop_after_epochs=stop_after_epochs, clip_max_norm=clip_max_norm,
                                    learning_rate=lr)
posterior = inference.build_posterior(density_estimator)
pickle.dump(posterior, open(f"{path}/{model}_posterior.pkl", "wb"))
