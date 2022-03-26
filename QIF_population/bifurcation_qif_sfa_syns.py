from pyauto import PyAuto

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
extract_le = False
n_dim = 4
n_params = 6
cont_idx = 2

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files", auto_dir='~/PycharmProjects/auto-07p')

# initial continuation in time
t_solutions, t_cont = a.run(e='qif_sfa_syns', c='ivp', ICP=14, DS=5e-3, DSMIN=1e-4, DSMAX=1.0, NMX=100000, name='t',
                            UZR={14: 100.0}, STOP={'UZ1'}, NDIM=n_dim, NPAR=n_params)

# continuation in the adaptation strength alpha
values = [0.05, 0.1, 0.2, 0.4]
alpha_solutions, alpha_cont = a.run(starting_point='UZ1', origin=t_cont, c='qif', ICP=3, UZR={3: values}, NDIM=n_dim,
                                    STOP=['UZ' + str(len(values))], DSMAX=0.001, NMX=6000, name='alpha', RL1=0.5,
                                    NPR=50)

# continuation in the excitability eta
values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
eta_solutions, eta_cont = a.run(starting_point='UZ3', origin=alpha_cont, c='qif', ICP=1, UZR={1: values}, NDIM=n_dim,
                                STOP=['UZ' + str(len(values))], DSMAX=0.001, NMX=8000, name='eta', RL1=6.0, RL0=-1.0,
                                NPR=50)

# principle continuation in J
#############################

# continue in J for each excitability eta
solutions_J = list()
i = 0
for point, point_info in eta_solutions.items():
    if 'UZ' in point_info['bifurcation']:
        solutions_J.append(a.run(starting_point=point, ICP=2, DSMAX=0.01, RL1=1.0, RL0=-20.0, NMX=4000,
                                 origin=eta_cont, name=f"J_{i}", NDIM=n_dim, DS=-1e-3, get_lyapunov_exp=extract_le))
        i += 1

# full 1D bifurcation analysis
##############################

# choose a continuation in Ja to run further continuations on
J_points, J_cont = solutions_J[cont_idx]

# limit cycle continuation of hopf bifurcations in J
hbs = ['HB' in s['bifurcation'] for s in J_points.values()]
if any(hbs):

    a.run(starting_point='HB1', c='qif2b', ICP=[2, 11], DSMAX=0.05, NMX=6000, origin=J_cont, name='J_hb1',
          NDIM=n_dim, NPR=20, STOP=['BP1', 'LP3'], get_lyapunov_exp=extract_le)

    if sum(hbs) > 1:

        eta_hb2_solutions, eta_hb2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[2, 11], DSMAX=0.05, NMX=6000,
                                                origin=J_cont, name='J_hb2', NDIM=n_dim, NPR=20, STOP=['BP1', 'LP3'],
                                                get_lyapunov_exp=extract_le)

################
# save results #
################

kwargs = dict()
kwargs['etas'] = values
fname = '../results/qif_sfa_syns.pkl'
a.to_file(fname, **kwargs)
