from pyauto import PyAuto
import numpy as np
import matplotlib.pyplot as plt

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
codim2 = True
period_mapping = False
n_grid_points = 100
n_dim = 8
n_params = 15
eta_cont_idx = 5

###################################
# parameter continuations in auto #
###################################

a = PyAuto("auto_files")

# initial continuations of connection strengths
J_0 = np.arange(12.0, 30.0, 2.0)
n_J = len(J_0)
Jei_solutions, Jei_cont = a.run(e='qif_stn_gpe', c='qif_stn_gpe', ICP=5, DSMAX=0.05, NMX=6000, NPAR=n_params,
                                bidirectional=True, name='Jei_0', NDIM=n_dim, RL0=0.0, UZR={5: J_0}, STOP={}, IPS=1,
                                ILP=1, NTST=200, NCOL=4, IAD=3, ISP=2, ISW=1, IPLT=0, NBC=0, NINT=0, NPR=500, MXBF=10,
                                IID=2, ITMX=8, ITNW=5, NWTN=3, JAC=0, EPSL=1e-07, EPSU=1e-07, EPSS=1e-05, IADS=1,
                                THL={11: 0.0}, THU={})

Jie_solutions, Jie_cont = a.run(starting_point='UZ6', ICP=6, DSMAX=0.05, NMX=6000, origin=Jei_cont, STOP=f'UZ{n_J}',
                                bidirectional=True, name='Jie_0', RL0=0.0, UZR={6: J_0})

# principle continuation in eta
###############################

# continue in eta for each adaptation rate alpha
solutions_eta = list()
i = 0
for point, point_info in Jie_solutions.items():
    if 'UZ' in point_info['bifurcation']:
        solutions_eta.append(a.run(starting_point=f'UZ{i+1}', ICP=1, NMX=8000, origin=Jie_cont, bidirectional=True,
                                   name=f'eta_{i}', RL0=-20.0, RL1=200.0))
        i += 1

# choose a continuation in eta to run further continuations on
eta_points, eta_cont = solutions_eta[eta_cont_idx]

fig, axes = plt.subplots(ncols=2)
ax = axes[0]
for i in range(n_J):
    ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_{i}', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E')

if codim1:

    # limit cycle continuation of hopf bifurcations in eta
    eta_hb_solutions, eta_hb_cont = a.run(starting_point='HB1', ICP=[1, 11], NMX=20000, origin=eta_cont, name='eta_hb',
                                          IPS=2, DSMAX=1.0)

    ax = axes[1]
    ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_{eta_cont_idx}', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E')
    ax = a.plot_continuation('PAR(1)', 'U(1)', cont=f'eta_hb', ax=ax, ignore=['BP'], line_color_stable='#148F77')
    plt.show()

    # limit cycle continuation of hopf bifurcation in beta
    beta_hb_solutions, beta_hb_cont = a.run(starting_point='HB1', ICP=[8, 11], NMX=7000, origin=eta_cont, IPS=2,
                                            RL0=-0.00001, name='beta_hb')

    # branch point continuation in beta
    beta_bp_solutions, beta_bp_cont = a.run(starting_point='BP1', ICP=[8, 11], origin=beta_hb_cont, ISW=-1,
                                            name='jee_hb')

    # limit cycle continuation of hopf bifurcation in alpha
    alpha_hb_solutions, alpha_hb_cont = a.run(starting_point='HB1', ICP=[7, 11], NMX=7000, origin=eta_cont,
                                              IPS=2, name='alpha_hb', RL0=-0.00001)

    if codim2:

        # continue the limit cycle borders in eta_i and beta
        #etai_beta_hb1_solutions, etai_beta_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[8, 2], DSMAX=0.05,
        #                                                    NMX=8000, bidirectional=True, origin=eta_cont, RL0=-0.001,
        #                                                    name='etai_beta_hb1', NDIM=n_dim, RL1=1.0)
        #etai_beta_hb2_solutions, etai_beta_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[8, 2], DSMAX=0.05,
        #                                                    NMX=8000, bidirectional=True, origin=eta_cont, RL0=-0.001,
        #                                                    name='etai_beta_hb2', NDIM=n_dim, RL1=1.0)

        # continue the stable limit cycle borders in eta and alpha
        #eta_alpha_hb1_solutions, eta_balpha_hb1_cont = a.run(starting_point='HB1', c='qif2', ICP=[7, 1], DSMAX=0.05,
        #                                                     NMX=8000, bidirectional=True, origin=eta_cont, RL0=-0.001,
        #                                                     name='eta_alpha_hb1', NDIM=n_dim, RL1=1.0)
        #eta_alpha_hb2_solutions, eta_alpha_hb2_cont = a.run(starting_point='HB2', c='qif2', ICP=[7, 1], DSMAX=0.05,
        #                                                    NMX=8000, bidirectional=True, origin=eta_cont, RL0=-0.001,
        #                                                    name='eta_alpha_hb2', NDIM=n_dim, RL1=1.0)

        # limit cycle continuation of hopf fold bifurcation in beta
        #beta_hb_solutions, beta_hb_cont = a.run(starting_point='LP2', c='qif', ICP=8, NDIM=n_dim, NPAR=n_params,
        #                                        DSMAX=0.01, NMX=10000, RL0=-0.001, RL1=1.0, name='beta_hb', ILP=0,
        #                                        origin=eta_hb_cont, EPSL=1e-6, EPSU=1e-6, EPSS=1e-4, NTST=400, IPS=2)

        # limit cycle continuation of hopf fold bifurcation in alpha
        #alpha_hb_solutions, alpha_hb_cont = a.run(starting_point='LP2', c='qif', ICP=7, NDIM=n_dim, NTST=400, ILP=0,
        #                                          NPAR=n_params, DSMAX=0.01, NMX=10000, RL0=-0.001, name='alpha_hb',
        #                                          RL1=1.0, origin=eta_hb_cont, EPSL=1e-6, EPSU=1e-6, EPSS=1e-4, IPS=2)

        pass

################
# save results #
################

fname = '../results/eic.pkl'
kwargs = dict()
a.to_file(fname, **kwargs)
