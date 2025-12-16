import pickle
import warnings
import torch
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sbi import utils as utils
from sbi.utils.user_input_checks import process_prior

# parameter definitions
#######################

# number of parameter samples to generate
n_samples = 1000000

# define directories
path = "/home/richard/data/sbi_organoids"
prior = None

# choose model
model = "pc"
op = "ik_full_op"

# free parameter bounds
param_bounds = {
    "C": (50.0, 200.0),
    "k": (0.5, 1.5),
    "Delta": (0.01, 10.0),
    "eta": (-5.0, 200.0),
    "kappa": (0.0, 100.0),
    "alpha": (0.0, 1.0),
    "g_a": (0.0, 100.0),
    "g_n": (0.0, 10.0),
    "b": (-10.0, 5.0),
    "U0": (0.0, 1.0),
    "tau_ca": (50.0, 500.0),
    "tau_w": (10.0, 100.0),
    "tau_u": (20.0, 200.0),
    "tau_x": (200.0, 2000.0),
    "tau_a": (1.0, 10.0),
    "tau_n": (15.0, 150.0),
    "tau_s": (0.5, 5.0),
    "psi": (50.0, 500.0),
    "theta": (0.0, 0.1),
    "noise_tau": (2.0, 200.0),
    "noise_scale": (0.0, 10.0)
}
param_keys = list(param_bounds.keys())
n_params = len(param_keys)

# create priors
if prior is None:

    # create uniform prior
    prior_min = [param_bounds[key][0] for key in param_keys]
    prior_max = [param_bounds[key][1] for key in param_keys]
    prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
    )

else:

    # load previous model fit
    prior = pickle.load(open(prior, "rb"))

# Check prior, simulator, consistency
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# generate parameter samples
theta = prior.sample((n_samples,))

# save parameters samples to file
pickle.dump({"theta": theta.cpu().numpy(), "parameters": param_keys, "bounds": param_bounds},
            open(f"{path}/ik_full_parameters.pkl", "wb"))
