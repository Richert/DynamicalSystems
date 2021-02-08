from pyrates.frontend import CircuitTemplate
import numpy as np
import matplotlib.pyplot as plt
from pyrates.utility.visualization import plot_network_graph

# parameter definition
dt = 1e-3
dts = 1e-1
cutoff = 0.0
T = 200.0 + cutoff
start = int((10 + cutoff)/dt)
dur = int(5/(0.6*dt))
steps = int(T/dt)
inp = np.zeros((steps, 1))
inp[start:start+dur] = 0.05

# model setup
path = "config/config_net/net"
template = CircuitTemplate.from_yaml(path).apply()
plot_network_graph(template)

model = template.compile(backend='numpy', solver='scipy', step_size=dt)

# simulation
results = model.run(simulation_time=T, step_size=dt, sampling_step_size=dts,
                    inputs={'RU/pg/u': inp},
                    outputs={'r_eu': 'EU/pg/p', 'r_us': 'US/pg/p', 'r_ru': 'RU/pg/p'}
                    )

results.plot()
plt.show()

# optimization
# targets = results