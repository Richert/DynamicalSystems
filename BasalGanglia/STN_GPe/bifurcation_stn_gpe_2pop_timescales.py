from pyrates.utility.pyauto import PyAuto
import sys
import numpy as np
import pandas as pd

"""Bifurcation analysis of STN-GPe model with two GPe populations (arkypallidal and prototypical) and 
gamma-distributed axonal delays and bi-exponential synapses."""

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 13
n_params = 22
a = PyAuto("auto_files", auto_dir=auto_dir)
kwargs = dict()
model = 'stn_gpe_2pop_timescales'
fname = f'../results/{model}_conts.pkl'
n_gridpoints = 40

#########################
# initial continuations #
#########################

# continuation in time
t_sols, t_cont = a.run(e=model, c='ivp', ICP=14, NMX=1000000, name='t', UZR={14: 1000.0}, STOP={'UZ1'},
                       NDIM=n_dim, NPAR=n_params)

starting_point = 'UZ1'
starting_cont = t_cont

# choose relative strength of GPe-p vs. GPe-a axons
kp_sols, kp_cont = a.run(starting_point=starting_point, c='qif', ICP=16, NPAR=n_params, name='k_p', NDIM=n_dim,
                         RL0=0.8, RL1=2.1, origin=starting_cont, NMX=2000, DSMAX=0.1, UZR={16: [2.0]}, STOP={})

starting_point = 'UZ1'
starting_cont = kp_cont

# choose relative strength of inter- vs. intra-population coupling inside GPe
ki_sols, ki_cont = a.run(starting_point=starting_point, c='qif', ICP=17, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.8, RL1=2.1, origin=starting_cont, NMX=2000, DSMAX=0.1, UZR={17: [1.5]}, STOP={})

starting_point = 'UZ1'
starting_cont = ki_cont

# preparation of healthy state
##############################

# continuation of eta_e
c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=1, NPAR=n_params,
                               name=f'eta_e:tmp', NDIM=n_dim, RL0=-1.0, RL1=10.0, origin=starting_cont,
                               NMX=8000, DSMAX=0.1, UZR={1: [4.0]})

starting_point = 'UZ1'
starting_cont = c2_b1_cont

# continuation of eta_p
c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                               name=f'eta_p:tmp', NDIM=n_dim, RL0=-1.0, RL1=10.0, origin=starting_cont,
                               NMX=4000, DSMAX=0.1, UZR={2: [3.145]})

starting_point = 'UZ1'
starting_cont = c2_b2_cont

# continuation of k_pe
c2_b3_sols, c2_b3_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=4, NDIM=n_dim,
                               NPAR=n_params, RL1=25.0, NMX=4000, DSMAX=0.05, name=f'k_pe:tmp',
                               UZR={4: [8.0]}, STOP=['UZ1'])

starting_point = 'UZ1'
starting_cont = c2_b3_cont

# continuation of pd-related parameters
#######################################

# continuations of k_gp for k_pe = 16.0
c2_b5_sols, c2_b5_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif', ICP=15, NDIM=n_dim,
                               NPAR=n_params, RL0=0.0, RL1=20.0, NMX=12000, DSMAX=0.05, name=f'k_gp:1',
                               bidirectional=True, UZR={15: [5.0]})

# 2D Hopf curve: tau_e x tau_p
tau_e = np.round(np.linspace(5.0, 20.0, n_gridpoints), decimals=2).tolist()
tau_p = np.round(np.linspace(15.0, 35.0, n_gridpoints), decimals=2).tolist()
c2_b5_2d1_sols, c2_b5_2d1_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[19, 20], NDIM=n_dim,
                                       NPAR=n_params, RL0=5.0, RL1=25.0, NMX=8000, DSMAX=0.1,
                                       name=f'tau_e/tau_p:hb1', bidirectional=True, UZR={19: tau_e})

tau_e_p_periods = np.zeros((len(tau_p), len(tau_e)))
i = 0
for s, s_info in c2_b5_2d1_sols.items():
    if 'UZ' in s_info['bifurcation']:
        i += 1
        s_tmp, _ = a.run(starting_point=f'UZ{i}', c='qif2b', ICP=[20, 11], UZR={20: tau_p}, get_period=True,
                         DSMAX=0.05, origin=c2_b5_2d1_cont, NMX=4000, NDIM=n_dim, NPAR=n_params, ILP=0, ISP=0, RL0=12.0,
                         RL1=36.0, STOP=['BP1', 'LP2'])
        for s2 in s_tmp.values():
            if 'UZ' in s2['bifurcation']:
                idx_c = np.argwhere(np.round(s2['PAR(19)'], decimals=2) == tau_e)
                idx_r = np.argwhere(np.round(s2['PAR(20)'], decimals=2) == tau_p)
                if s2['period'] > tau_e_p_periods[idx_r, idx_c]:
                    tau_e_p_periods[idx_r, idx_c] = 1e3/s2['period']
kwargs['tau_e_p_periods'] = tau_e_p_periods
kwargs['tau_e'] = tau_e
kwargs['tau_p'] = tau_p

# 2D Hopf curve: tau_ampa_d x tau_gabaa_d
tau_ampa = np.round(np.linspace(3.0, 10.0, n_gridpoints), decimals=2).tolist()
tau_gabaa = np.round(np.linspace(5.0, 35.0, n_gridpoints), decimals=2).tolist()
c2_b5_2d2_sols, c2_b5_2d2_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[21, 22], NDIM=n_dim,
                                       NPAR=n_params, RL0=3.0, RL1=10.0, NMX=8000, DSMAX=0.05,
                                       name=f'tau_ampa_d/tau_gabaa_d:hb1', bidirectional=True, UZR={21: tau_ampa})

tau_ampa_gabaa_periods = np.zeros((len(tau_gabaa), len(tau_ampa)))
i = 0
for s, s_info in c2_b5_2d2_sols.items():
    if 'UZ' in s_info['bifurcation']:
        i += 1
        s_tmp, _ = a.run(starting_point=f'UZ{i}', c='qif2b', ICP=[22, 11], UZR={22: tau_gabaa}, get_period=True,
                         DSMAX=0.05, origin=c2_b5_2d2_cont, NMX=4000, NDIM=n_dim, NPAR=n_params, ILP=0, ISP=0, RL0=5.0,
                         RL1=35.0, STOP=['BP1', 'LP2'])
        for s2 in s_tmp.values():
            if 'UZ' in s2['bifurcation']:
                idx_c = np.argwhere(np.round(s2['PAR(21)'], decimals=2) == tau_ampa)
                idx_r = np.argwhere(np.round(s2['PAR(22)'], decimals=2) == tau_gabaa)
                if s2['period'] > tau_ampa_gabaa_periods[idx_r, idx_c]:
                    tau_ampa_gabaa_periods[idx_r, idx_c] = 1e3/s2['period']
kwargs['tau_ampa_gabaa_periods'] = tau_ampa_gabaa_periods
kwargs['tau_ampa'] = tau_ampa
kwargs['tau_gabaa'] = tau_gabaa

# 2D Hopf curve: tau_gabaa_d x tau_stn
tau_stn = np.round(np.linspace(1.9, 3.0, n_gridpoints), decimals=2).tolist()
tau_gabaa2 = np.round(np.linspace(4.0, 30.0, n_gridpoints), decimals=2).tolist()
c2_b5_2d3_sols, c2_b5_2d3_cont = a.run(starting_point='HB1', origin=c2_b5_cont, c='qif2', ICP=[22, 18], NDIM=n_dim,
                                       NPAR=n_params, RL0=4.0, RL1=30.0, NMX=8000, DSMAX=0.1,
                                       name=f'tau_gabaa_d/tau_stn:hb1', bidirectional=True, UZR={18: tau_stn})

tau_stn_gabaa_periods = np.zeros((len(tau_gabaa2), len(tau_stn)))
i = 0
for s, s_info in c2_b5_2d3_sols.items():
    if 'UZ' in s_info['bifurcation']:
        i += 1
        s_tmp, _ = a.run(starting_point=f'UZ{i}', c='qif2b', ICP=[22, 11], UZR={22: tau_gabaa2}, get_period=True,
                         DSMAX=0.05, origin=c2_b5_2d3_cont, NMX=4000, NDIM=n_dim, NPAR=n_params, ILP=0, ISP=0, RL0=4.0,
                         RL1=30.0, STOP=['BP1', 'LP2'])
        for s2 in s_tmp.values():
            if 'UZ' in s2['bifurcation']:
                idx_c = np.argwhere(np.round(s2['PAR(18)'], decimals=2) == tau_stn)
                idx_r = np.argwhere(np.round(s2['PAR(22)'], decimals=2) == tau_gabaa2)
                if s2['period'] > tau_stn_gabaa_periods[idx_r, idx_c]:
                    tau_stn_gabaa_periods[idx_r, idx_c] = 1e3/s2['period']
kwargs['tau_stn_gabaa_periods'] = tau_stn_gabaa_periods
kwargs['tau_stn'] = tau_stn
kwargs['tau_gabaa2'] = tau_gabaa2

# save results
a.to_file(fname, **kwargs)

# plotting
import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=3, nrows=2)

ax1 = axes[0, 0]
ax1 = a.plot_continuation('PAR(19)', 'PAR(20)', cont='tau_e/tau_p:hb1', line_style_unstable='solid', ignore=['UZ'],
                          ax=ax1)
ax1.set_xlabel(r'$\tau_e$')
ax1.set_ylabel(r'$\tau_p$')

ax2 = axes[0, 1]
ax2 = a.plot_continuation('PAR(21)', 'PAR(22)', cont='tau_ampa_d/tau_gabaa_d:hb1', line_style_unstable='solid',
                          ignore=['UZ'], ax=ax2)
ax2.set_xlabel(r'$\tau_{ampa}$')
ax2.set_ylabel(r'$\tau_{gabaa}$')

ax3 = axes[0, 2]
ax3 = a.plot_continuation('PAR(18)', 'PAR(22)', cont='tau_gabaa_d/tau_stn:hb1', line_style_unstable='solid',
                          ignore=['UZ'], ax=ax3)
ax3.set_xlabel(r'$\tau_{stn}$')
ax3.set_ylabel(r'$\tau_{gabaa}$')

ax4 = axes[1, 0]
df1 = pd.DataFrame(tau_e_p_periods, index=tau_p, columns=tau_e)
ax4 = a.plot_heatmap(df1, ax=ax4, cmap='magma', mask=df1.values == 0)
ax4.invert_yaxis()

ax5 = axes[1, 1]
df2 = pd.DataFrame(tau_ampa_gabaa_periods, index=tau_gabaa, columns=tau_ampa)
ax5 = a.plot_heatmap(df2, ax=ax5, cmap='magma', mask=df2.values == 0)
ax5.invert_yaxis()

ax6 = axes[1, 2]
df3 = pd.DataFrame(tau_stn_gabaa_periods, index=tau_gabaa2, columns=tau_stn)
ax6 = a.plot_heatmap(df3, ax=ax6, cmap='magma', mask=df3.values == 0)
ax6.invert_yaxis()

plt.tight_layout()
plt.show()
