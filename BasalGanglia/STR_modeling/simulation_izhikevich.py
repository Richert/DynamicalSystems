from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
import pickle

# load model
ik = CircuitTemplate.from_yaml("config/ik/ik")

# adjust parameters
ik.update_var(node_vars={'p/ik_op/Delta': 0.2})

# define simulation
T = 2100.0
dt = 1e-4
dts = 1e-2
steps = int(np.round(T/dt))
start = int(np.round(600/dt))
stop = int(np.round(1600/dt))
inp = np.zeros((steps,)) + 60.0
inp[start:stop] -= 15.0

# run simulation
res = ik.run(simulation_time=T, step_size=dt, sampling_step_size=dts, solver='scipy',
             outputs={'r': 'p/ik_op/r', 'v': 'p/ik_op/v', 'u': 'p/ik_op/u'},
             inputs={'p/ik_op/I_ext': inp}, cutoff=100.0, method='LSODA', atol=1e-6, rtol=1e-5)

# plotting
fig, axes = plt.subplots(nrows=3, figsize=(12, 9))
axes[0].plot(res['r'])
axes[0].set_ylabel('r')
axes[1].plot(res['v'])
axes[1].set_ylabel('v')
axes[2].plot(res['u'])
axes[2].set_ylabel('u')
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/izhikevich_sim_exc1.p", "wb"))
