from pyrates.utility import PyAuto
import numpy as np

#########################################
# configs, descriptions and definitions #
#########################################

# problem description
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
codim1 = True
codim2 = False
period_mapping = False
n_grid_points = 100
n_dim = 4
n_params = 6
eta_cont_idx = 1

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuation in the adaptation strength alpha
alpha_0 = [0.05]
alpha_solutions, alpha_cont = a.run(e='qif_alpha_mult_hc', c='qifa', ICP=5, UZR={5: alpha_0}, NDIM=n_dim,
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
        solutions_eta.append(a.run(starting_point=point, ICP=1, DSMAX=0.001, RL1=0.0, RL0=-12.0, NMX=10000,
                                   origin=alpha_cont, bidirectional=False, name=f"eta_{i}", NDIM=n_dim,
                                   UZR={1: [-5.6810471]}, STOP=['UZ4']))
        i += 1

# choose a continuation in eta to run further continuations on
eta_points, eta_cont = solutions_eta[eta_cont_idx]

if codim1:

    # limit cycle continuation of hopf bifurcations in eta
    eta_hb2_solutions, eta_hb2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[1, 11], DSMAX=0.5, NMX=100000,
                                            origin=eta_cont, name='eta_hb2', NDIM=n_dim, UZR={11: {1000.0}}, STOP={'BP20'},
                                            get_timeseries=True)

    a.plot_timeseries('U(1)', points=['LP1', 'BP1', 'BP15'], linespecs=[{'color': (0,0,0,1)}, {'color': (0,1,0,1)}, {'color': (0,0,1,1)}],
                      cont='eta_hb2')
    import matplotlib.pyplot as plt
    plt.show()

    # homoclinic bifurcation analysis
    eta_ho_solutions, eta_ho_cont = a.run(starting_point='UZ3', origin=eta_cont, c='kpr.1', ICP=[1, 11])

    # continuation in eta and alpha
    ###############################

    if codim2:

        # continue the limit cycle borders in alpha and eta
        eta_alpha_hb2_solutions, eta_alpha_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[1, 3], DSMAX=0.001,
                                                            NMX=8000, bidirectional=True, origin=eta_cont,
                                                            name='eta_alpha_hb2', NDIM=n_dim)

        # continue the fold borders in alpha and eta
        eta_alpha_lp1_solutions, eta_alpha_lp1_cont = a.run(starting_point='LP1', c='qif2', ICP=[1, 3], DSMAX=0.001,
                                                            NMX=8000, origin=eta_cont, name='eta_alpha_lp1', NDIM=n_dim,
                                                            bidirectional=True)

        # continue the first and second fold bifurcation of the limit cycle
        alphas = np.round(np.linspace(0., 0.2, 100)[::-1], decimals=5).tolist()
        eta_alpha_lp2_solutions, eta_alpha_lp2_cont = a.run(starting_point='LP1', c='qif3', ICP=[1, 11, 3],
                                                            NMX=8000, DSMAX=0.01, origin=eta_hb2_cont,
                                                            bidirectional=True, name='eta_alpha_lp2', NDIM=n_dim,
                                                            UZR={3: alphas}, STOP={})
        eta_alpha_lp3_solutions, eta_alpha_lp3_cont = a.run(starting_point='LP2', c='qif3', ICP=[1, 11, 3],
                                                            NMX=8000, DSMAX=0.01, origin=eta_hb2_cont,
                                                            bidirectional=True, name='eta_alpha_lp3', NDIM=n_dim,
                                                            UZR={3: alphas}, STOP={})

        if period_mapping:

            # extract limit cycle periods in eta-alpha plane
            etas = np.round(np.linspace(-6.5, -2.5, 100), decimals=4).tolist()
            period_solutions = np.zeros((len(alphas), len(etas)))
            for s, s_info in eta_alpha_lp3_solutions.items():
                if np.round(s_info['PAR(3)'], decimals=5) in alphas:
                    s_tmp, _ = a.run(starting_point=s, c='qif', ICP=[1, 11], UZR={1: etas}, STOP={}, EPSL=1e-6,
                                     EPSU=1e-6, EPSS=1e-4, ILP=0, ISP=0, IPS=2, get_period=True, DSMAX=0.0002,
                                     origin=eta_alpha_lp3_cont, NMX=40000, DS='-', THL={11: 0.0})
                    for s2 in s_tmp.v1():
                        if np.round(s2['PAR(1)'], decimals=4) in etas:
                            idx_c = np.argwhere(np.round(s2['PAR(1)'], decimals=4) == etas)
                            idx_r = np.argwhere(np.round(s2['PAR(3)'], decimals=5) == alphas)
                            if s2['period'] > period_solutions[idx_r, idx_c]:
                                period_solutions[idx_r, idx_c] = s2['period']

################
# save results #
################

fname = '../results/alpha_mult.pkl'
kwargs = dict()
if period_mapping:
    kwargs['tau_e_p_periods'] = period_solutions
a.to_file(fname, **kwargs)
