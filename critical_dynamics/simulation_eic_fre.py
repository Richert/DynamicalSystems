from pyrates import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt


# initialize network
eic = CircuitTemplate.from_yaml("config/ik_mf/eic")

# simulation parameters
T = 2500.0
cutoff = 500.0
dt = 1e-3
dts = 1e-1
inp_lts = np.zeros((int(T/dt),)) + 80.0
inp_lts[int(1000/dt):int(2000/dt)] += 40.0
inp_rs = np.zeros_like(inp_lts) + 50.0

# perform simulation
res = eic.run(T, dt, sampling_step_size=dts, solver="scipy", method="RK45", rtol=1e-6, atol=1e-6,
              inputs={"lts/lts_op/I_ext": inp_lts, "rs/rs_op/I_ext": inp_rs},
              outputs={"r_e": "rs/rs_op/r", "r_i": "lts/lts_op/r"})

# plotting
fig, ax = plt.subplots(figsize=(12, 4))
res.plot(ax=ax)
plt.show()
