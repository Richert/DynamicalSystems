from pyrates.utility.pyauto import PyAuto, continue_period_doubling_bf, fractal_dimension, get_from_solutions

#########################################
# configs, descriptions and definitions #
#########################################

# problem description
#####################

"""
Performs continuation of extended Montbrio population given the initial condition:

 U1: r_0 = 0.114741, 
 U2: v_0 = -2.774150,
 U3: e_0 = 0.0,
 U4: a_0 = 0.0

with parameters:

 PAR1: eta = -10.0
 PAR2: J = 15.0
 PAR3: alpha = 0.0
 PAR4: tau = 1.0
 PAR5: D = 2.0

"""

# configuration
###############

# configuration
codim1 = True
codim2 = True
n_grid_points = 100
n_dim = 3
n_params = 6
eta_cont_idx = 2

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuation in the adaptation strength alpha
alpha_0 = [0.0125, 0.025, 0.05, 0.1, 0.2]
alpha_solutions, alpha_cont = a.run(e='qif_exp_mult', c='qif', ICP=3, UZR={3: alpha_0}, NDIM=n_dim,
                                    STOP=['UZ' + str(len(alpha_0))], DSMAX=0.005, NMX=4000, name='s0')

# principle continuation in eta
###############################

# continue in eta for each adaptation rate alpha
solutions_eta = list()
solutions_eta.append(a.run(starting_point='EP1', ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, NMX=2000, origin=alpha_cont,
                           bidirectional=True, name='eta_0', NDIM=n_dim))

i = 1
for point, point_info in alpha_solutions.items():
    if 'UZ' in point_info['bifurcation']:
        solutions_eta.append(a.run(starting_point=point, ICP=1, DSMAX=0.005, RL1=0.0, RL0=-12.0, NMX=4000,
                                   origin=alpha_cont, bidirectional=True, name=f"eta_{i}", NDIM=n_dim))
        i += 1

# choose a continuation in eta to run further continuations on
eta_points, eta_cont = solutions_eta[eta_cont_idx]

if codim1:

    # limit cycle continuation of hopf bifurcations in eta
    eta_hb1_solutions, eta_hb1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[1, 11], DSMAX=0.05, NMX=3000,
                                            origin=eta_cont, name='eta_hb1', NDIM=n_dim)
    eta_hb2_solutions, eta_hb2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[1, 11], DSMAX=0.05, NMX=2500,
                                            origin=eta_cont, name='eta_hb2', NDIM=n_dim, get_lyapunov_exp=True,
                                            get_timeseries=True, NPR=20, STOP={'BP1'})

    # continue the period doubling bifurcations in eta that we found above
    pds, a = continue_period_doubling_bf(solution=eta_hb2_solutions, continuation=eta_hb2_cont, pyauto_instance=a,
                                         c='qif2b', ICP=[1, 11], NMX=2500, DSMAX=0.05, NTST=800, ILP=0, NDIM=n_dim,
                                         get_timeseries=True, NPR=20, STOP={'BP1'})
    pds.append('eta_hb2')

    # extract Lyapunov exponents from solutions (RICHARD: moved this part to after the period doubling continuations)
    chaos_analysis_hb2 = dict()

    # # RICHARD: iterate over the names of all limit cycle continuations stored in pds
    # for s in pds:
    #     # RICHARD: extract the solution curve for a given continuation
    #     sol_tmp = a.get_summary(cont=s)
    #
    #     # RICHARD: extract eta and lyapunov exponent from each solution on the solution curve (returned as a list of lists by get_from_solutions, which I import above)
    #     data = get_from_solutions(keys=['PAR(1)', 'lyapunov_exponents'], solutions=sol_tmp)
    #     etas = [d[0] for d in data]
    #     lyapunovs = [d[1] for d in data]
    #
    #     # create a dictionary with point as key, save eta and Lyapunov exponents in it
    #     chaos_analysis_hb2[s] = dict()
    #     chaos_analysis_hb2[s]['eta'] = etas
    #     chaos_analysis_hb2[s]['lyapunov_exponents'] = lyapunovs
    #
    #     # compute fractal dimension of attractor at each solution point
    #     chaos_analysis_hb2[s]['fractal_dim'] = [fractal_dimension(lp) for lp in lyapunovs]

    # continuation in eta and alpha
    ###############################

    if codim2:
        # continue the limit cycle borders in alpha and eta
        eta_alpha_hb2_solutions, eta_alpha_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[1, 3], DSMAX=0.01,
                                                            NMX=2000, bidirectional=True, origin=eta_cont,
                                                            name='eta_alpha_hb2', NDIM=n_dim, NPR=10)

        # continue the first period doubling of the limit cycle
        eta_alpha_pd_solutions, eta_alpha_pd_cont = a.run(starting_point='PD1', c='qif3', ICP=[1, 3, 11],
                                                          NMX=3000, DSMAX=0.05, origin=eta_hb2_cont,
                                                          bidirectional=True, name='eta_alpha_pd', NTST=600,
                                                          NDIM=n_dim, ILP=0, NPR=10)

################
# save results #
################

fname = '../results/exp_mult.pkl'

kwargs = {}
if codim1:
    kwargs['pd_solutions'] = pds
    #kwargs['chaos_analysis_hb2'] = chaos_analysis_hb2
a.to_file(fname, **kwargs)
