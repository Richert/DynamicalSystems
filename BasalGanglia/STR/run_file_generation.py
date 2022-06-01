from pyrates import CircuitTemplate

# choose neuron type
neuron_type = 'spn_d1'

# redefine model parameters
node_vars = {}

# load model template
template = CircuitTemplate.from_yaml(f'config/model_def/{neuron_type}')

# update template parameters
template.update_var(node_vars=node_vars)

# set pyrates-specific parameters
dt = 1e-4
backend = 'fortran'
kwargs = {'auto': True, 'vectorize': False, 'float_precision': 'float64', }

# generate model equations
template.get_run_func(func_name=f'{neuron_type}_run', file_name=f'config/{neuron_type}_equations',
                      step_size=dt, backend=backend, solver='scipy', **kwargs)
