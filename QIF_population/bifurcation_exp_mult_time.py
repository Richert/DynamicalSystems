from pyrates.utility.pyauto import PyAuto, continue_period_doubling_bf, fractal_dimension, get_from_solutions

# configuration
###############

# configuration
codim1 = True
codim2 = True
n_grid_points = 100
n_dim = 3
n_params = 6
eta_cont_idx = 1

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuation in the adaptation strength alpha
alpha_0 = [0.05]
alpha_solutions, alpha_cont = a.run(e='qif_exp_mult', c='qif', ICP=3, UZR={3: alpha_0}, NDIM=n_dim,
                                    STOP=['UZ1'], DSMAX=0.005, NMX=10000, name='s0')

# principle continuation in eta
###############################

# continue in eta for adaptation rate alpha

# what are we doing here? before iterating over alpha?
# RICHARD: since we cannot create a UZ point for alpha = 0.0, we perform the first continuation outside of the for loop
# (the continuation already starts at alpha = 0.0 and thus cannot set a user point there, since the continuation never passes that point again)
# since you want to do this only for a single alpha (alpha = 0.05, I simplified things a little bit below

# etas = [-5.15, -5.21, -5.3] # save these points
# etas = [-5.15, -5.25, -5.35]
etas = [-5.15, -5.22, -5.3]

eta_solutions, eta_cont = a.run(starting_point='UZ1', ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, NMX=10000,
                                origin=alpha_cont,
                                bidirectional=True, name='eta_0', NDIM=n_dim, STOP=['HB2'])

# we now jump on the limit cycle
eta_hb1_solutions, eta_hb1_cont = a.run(starting_point='HB2', c='qif2b', ICP=[1, 11], DSMAX=0.05, NMX=2700,
                                        origin=eta_cont, name='eta_hb1', NDIM=n_dim, UZR={1: etas}, STOP=['BP1', 'LP9'],
                                        get_lyapunov_exp=True)

# continue in time for the three points in eta
# RICHARD: in the if condition, you checked in your script whether 'UZ_eta' was in point_info['bifurcation']. However, auto only uses 'UZ#',
# where # is an integer number for the #th user specified point. I changed the if condition accordingly
# also, I added '_t' to the naming of the solution branch, to indicate the difference to the continuation in eta performed above.
chaos_analysis = {}

i = 1
for point, point_info in eta_hb1_solutions.items():
    if 'UZ' in point_info['bifurcation']:
        sol_time, auto_obj_time = a.run(starting_point=point, ICP=14, c='ivp', DSMAX=0.005, NPR=100, DSMIN=1e-9,
                                        NMX=10000000,
                                        origin=eta_hb1_cont, name=f"eta_{i}_t", NDIM=n_dim, UZR={14: [200]},
                                        STOP=['UZ1'])

        print('here')

        # Lyapunov exponents and dimensionality of attractor
        chaos_analysis[f"eta_{i}_t"] = {}

        # extract eta and lyapunov exponent from each solution on the solution curve (returned as a list of lists by get_from_solutions, which I import above)
        data = get_from_solutions(keys=['PAR(14)', 'lyapunov_exponents'], solutions=sol_time)
        time = [d[0] for d in data]
        lyapunovs = [d[1] for d in data]

        # create a dictionary with point as key, save eta and Lyapunov exponents in it
        chaos_analysis[f"eta_{i}_t"]['time'] = time
        chaos_analysis[f"eta_{i}_t"]['lyapunov_exponents'] = lyapunovs

        # compute fractal dimension of attractor at each solution point
        chaos_analysis[f"eta_{i}_t"]['fractal_dim'] = [fractal_dimension(lp) for lp in lyapunovs]

        i += 1

fname = '../results/exp_mult_strange_attractor.pkl'

kwargs = {}
kwargs['solutions_time'] = solutions_time
kwargs['etas'] = etas
kwargs['chaos_analysis'] = chaos_analysis
a.to_file(fname, **kwargs)
