import numpy as np
import matplotlib.pyplot as plt
from pyauto import PyAuto

pa = PyAuto(auto_dir='auto_files')

b, s = pa.run(variables=None, params=None, get_lyapunov_exp=True, get_period=True, e='qif', c='qif', ICP=1,
              DSMAX=0.005, NMX=3000)
b2, s2 = pa.run(variables=['U(1)', 'U(2)'], params=['PAR(1)', 'PAR(2)'], get_lyapunov_exp=True, get_period=True,
                c='qif2', ICP=[1, 2], DSMAX=0.005, NMX=4000, starting_point=(b, 'LP1'))

fig, axes = plt.subplots(ncols=2)
pa.plot_continuation('PAR(1)', 'U(1)', b, linewidths=1.0, colors='b', default_size=100, ax=axes[0])
pa.plot_continuation('PAR(1)', 'PAR(2)', b2, linewidths=1.0, colors='r', default_size=100, ax=axes[1], ignore=['LP'],
                     line_style_unstable='solid')
plt.show()
