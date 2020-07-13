"""
In this script, simple numerical simulations of the model behavior in time are performed for a single model
parameterization. The underlying model is composed of the macroscopic dynamics of a all-to-all coupled QIF
population with synaptic depression described by a convolution with a mono-exponential function:

.. math::

    \\tau \\dot r &= \\frac{\\Delta}{\\pi\\tau} + 2 r v, \n
    \\tau \\dot v &= v^2 +\\bar\\eta + I(t) + J r \\tau - A - (\\pi r \\tau)^2, \n
    \\tau_A \\dot A &= \\alpha r - \\frac{A}{\\tau_A},

where the evolution equations for :math:`A` and :math:`B` express a convolution of :math:`r` with an alpha kernel, with
adaptation strength :math:`\\alpha` and time constant :math:`\\tau_A`.

In the sections below, we will demonstrate for each model how to load the model template into pyrates, perform
simulations with it and visualize the results.

References
----------

.. [1] E. Montbrió, D. Pazó, A. Roxin (2015) *Macroscopic description for networks of spiking neurons.* Physical
       Review X, 5:021028, https://doi.org/10.1103/PhysRevX.5.021028.

.. [2] R. Gast, H. Schmidt, T.R. Knösche (2020) *A Mean-Field Description of Bursting Dynamics in Spiking Neural
       Networks with Short-Term Adaptation.* Neural Computation (in press).

"""

from pyrates.frontend import CircuitTemplate
from matplotlib.pyplot import show
qif_circuit = CircuitTemplate.from_yaml("model_templates.montbrio.simple_montbrio.QIF_sd_exp"
                                        ).apply(node_values={'p/Op_sd_exp/eta': -5.201,
                                                             'p/Op_sd_exp/alpha': 0.05})
qif_compiled = qif_circuit.compile(backend='numpy', step_size=1e-4, solver='scipy')
results = qif_compiled.run(simulation_time=960.0, outputs={'r': 'p/Op_sd_exp/r'}, method='RK45')
results.plot()
show()
