from pyrates.utility.pyauto import PyAuto

# configuration
###############

# configuration
n_grid_points = 100
n_dim = 3
n_params = 6
eta_cont_idx = 1

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuation in the adaptation strength alpha
alpha_0 = [0.7]
alpha_solutions, alpha_cont = a.run(e='qif_exp_add', c='qif', ICP=3, UZR={3: alpha_0}, NDIM=n_dim,
                                    STOP=['UZ1'], DSMAX=0.005, NMX=10000, name='s0')

# principle continuation in eta
###############################

# continue in eta for adaptation rate alpha
eta_solutions, eta_cont = a.run(starting_point='UZ1', ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, NMX=10000,
                                origin=alpha_cont,
                                bidirectional=True, name='eta_0', NDIM=n_dim, STOP=['HB2'])

# we now jump on the limit cycle
etas = [-3.9, -3.931, -3.962]  # save these points
eta_hb2_solutions, eta_hb2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[1, 11], DSMAX=0.05, NMX=2000,
                                        origin=eta_cont, name='eta_hb1', NDIM=n_dim, UZR={1: etas}, STOP=['BP1'])

# extract Lyapunov exponents and fractal dimension of attractor per eta
chaos_analysis = {}
chaos_analysis['eta'] = []
chaos_analysis['lyapunov_exponents'] = []
chaos_analysis['fractal_dim'] = []

# manually get solutions
for key, value in eta_hb2_solutions.items():
    if value['bifurcation'] == 'UZ':
        chaos_analysis['eta'].append(value['PAR(1)'])
        #chaos_analysis['lyapunov_exponents'].append(value['lyapunov_exponents'])
        #chaos_analysis['fractal_dim'].append(fractal_dimension(value['lyapunov_exponents']))

# continue in time for the three points in eta
solutions_time = []
i = 1
for point, point_info in eta_hb2_solutions.items():
    if 'UZ' in point_info['bifurcation']:
        sol_time, auto_obj_time = a.run(starting_point=point, ICP=14, c='ivp', DSMAX=0.005, DSMIN=1e-9, NPR=100,
                                        NMX=10000000, origin=eta_hb2_cont, name=f"eta_{i}_t", NDIM=n_dim,
                                        UZR={14: [1200]}, STOP=['UZ1'])

        solutions_time.append(sol_time)

        i += 1

fname = '../results/exp_add_time.pkl'

kwargs = {}
kwargs['solutions_time'] = solutions_time
kwargs['etas'] = etas
kwargs['chaos_analysis'] = chaos_analysis
a.to_file(fname, **kwargs)
