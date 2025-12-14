import pickle
from custom_functions import *
from pyrates import CircuitTemplate
from numba import njit
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# parameter definitions
#######################

# choose device
device = "cpu"

# define directories
path = "/home/richard/data/sbi_organoids"

# choose model
model = "pc"
op = "ik_full_op"

# simulation parameters
T = 60000.0
cutoff = 0.0
dt = 1e-3

# model parameters
C = 100.0
k = 0.7
v_r = -70.0
v_t = -45.0
Delta = 1.0
eta = 85.0
b = -2.0
kappa = 5.0
U0 = 0.6
alpha = 0.0
psi = 300.0
theta = 0.02
g_a = 30.0
g_n = 0.1
g_g = 0.0
tau_w = 50.0
tau_ca = 250.0
tau_u = 100.0
tau_x = 700.0
tau_a = 5.0
tau_n = 150.0
tau_g = 10.0
tau_s = 1.0
node_vars = {
    'C': C, 'k': k, 'v_r': v_r, 'v_t': v_t, 'Delta': Delta, 'eta': eta, 'kappa': kappa, 'alpha': alpha,
    'g_a': g_a, 'g_n': g_n, 'g_g': g_g, 'b': b, 'U0': U0, 'tau_ca': tau_ca, 'tau_w': tau_w, 'tau_u': tau_u,
    'tau_x': tau_x, 'tau_a': tau_a, 'tau_n': tau_n, 'tau_g': tau_g, 'tau_s': tau_s, 'psi': psi, 'theta': theta
}

# initialize model
##################

# initialize model template and set fixed parameters
template = CircuitTemplate.from_yaml(f"config/ik_mf/{model}")
template.update_var(node_vars={f"p/{op}/{key}": val for key, val in node_vars.items()})

# generate run function
func, args, arg_keys, _ = template.get_run_func(f"{model}_vectorfield", step_size=dt, backend="numpy", solver="scipy",
                                                float_precision="float32", vectorize=False,
                                                inputs={f"p/{op}/I_ext": np.zeros(int(T/dt),)})
# func_jit = njit(func)
# func_jit(*args)

# save model
pickle.dump({"func": func, "args": args, "arg_keys": arg_keys}, open(f"{path}/ik_full_model.pkl", "wb"))
