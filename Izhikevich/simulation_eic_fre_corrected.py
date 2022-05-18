from pyrecu import RNN
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'
from numpy import pi, sqrt, arctan
from numba import njit


@njit(fastmath=True)
def correct_input(inp: float, k: float, v_r: float, v_t: float, v_spike: float, v_reset: float, g_ampa: float,
                  s_ampa: float, E_ampa: float, g_gaba: float, s_gaba: float, E_gaba: float, u: float):
    alpha = v_r + v_t + (g_ampa*s_ampa + g_gaba*s_gaba)/k
    mu = 4*(v_r*v_t + (inp - u + g_ampa*s_ampa*E_ampa + g_gaba*s_gaba*E_gaba)/k) - alpha**2
    if mu > 0:
        mu_sqrt = sqrt(mu)
        inp_c = pi**2*k*mu/(4*(arctan((2*v_spike-alpha)/mu_sqrt) - arctan((2*v_reset-alpha)/mu_sqrt))**2)
        inp_c += k*alpha**2/4 + u
        inp_c -= k*v_r*v_t + g_ampa*s_ampa*E_ampa + g_gaba*s_gaba*E_gaba
    else:
        inp_c = inp
    return inp_c


def eic_run(y, N, I_ext_input, v_t, v_r, k, g_gabaa, g_ampa, Delta, C, E_gabaa, E_ampa, v_z, v_p, b, a, d, tau_ampa,
            tau_gabaa, v_t_0, v_r_0, k_0, g_gabaa_0, g_ampa_0, Delta_0, C_0, E_gabaa_0, E_ampa_0, v_z_0, v_p_0,
            b_0, a_0, d_0, tau_ampa_0, tau_gabaa_0, weight, weight_0, weight_1, weight_2, dt):

    r = y[0]
    v = y[1]
    u = y[2]
    s_ampa = y[3]
    s_gabaa = y[4]
    r_0 = y[5]
    v_0 = y[6]
    u_0 = y[7]
    s_ampa_0 = y[8]
    s_gabaa_0 = y[9]
    r_e = r * weight
    r_i = r_0 * weight_0
    r_e_0 = r * weight_1
    r_i_0 = r_0 * weight_2
    I_ext = correct_input(I_ext_input[0], k, v_r, v_t, v_p, v_z, g_ampa, s_ampa, E_ampa, g_gabaa, s_gabaa, E_gabaa, u)
    I_ext_0 = correct_input(I_ext_input[1], k_0, v_r_0, v_t_0, v_p_0, v_z_0, g_ampa_0, s_ampa_0, E_ampa_0, g_gabaa_0,
                            s_gabaa_0, E_gabaa_0, u_0)

    dy = [
        (r * (-g_ampa * s_ampa - g_gabaa * s_gabaa + k * (2.0 * v - v_r - v_t)) + Delta * k ** 2 * (v - v_r * (2)) / (
                    pi * C)) / C,
        (-pi ** 2 * C ** 2 * r ** 2 / k + I_ext + g_ampa * s_ampa * (
                    E_ampa - v) + g_gabaa * s_gabaa * (E_gabaa - v) + k * v * (v - v_r - v_t) + k * v_r * v_t - u) / C,
        a * (b * (v - v_r) - u) + d * r,
        r_e - s_ampa / tau_ampa,
        r_i - s_gabaa / tau_gabaa,
        (r_0 * (-g_ampa_0 * s_ampa_0 - g_gabaa_0 * s_gabaa_0 + k_0 * (
                    2.0 * v_0 - v_r_0 - v_t_0)) + Delta_0 * k_0 ** 2 * (v_0 - v_r_0) / (pi * C_0)) / C_0,
        (-pi ** 2 * C_0 ** 2 * r_0 ** 2 / k_0 + I_ext_0 + g_ampa_0 * s_ampa_0 * (E_ampa_0 - v_0) + g_gabaa_0 * s_gabaa_0 * (
                             E_gabaa_0 - v_0) + k_0 * v_0 * (v_0 - v_r_0 - v_t_0) + k_0 * v_r_0 * v_t_0 - u_0) / C_0,
        a_0 * (b_0 * (v_0 - v_r_0) - u_0) + d_0 * r_0,
        r_e_0 - s_ampa_0 / tau_ampa_0,
        r_i_0 - s_gabaa_0 / tau_gabaa_0,
    ]

    for i, d in enumerate(dy):
        y[i] += dt*d

    return y


# define model parameters
#########################

# RS neuron parameters
Ce = 100.0   # unit: pF
ke = 0.7  # unit: None
ve_r = -60.0  # unit: mV
ve_t = -40.0  # unit: mV
ve_spike = 40.0  # unit: mV
ve_reset = -60.0  # unit: mV
Delta_e = 1.0  # unit: mV
de = 10.0
ae = 0.03
be = -2.0

# IB neuron parameters
Ci = 20.0   # unit: pF
ki = 1.0  # unit: None
vi_r = -55.0  # unit: mV
vi_t = -40.0  # unit: mV
vi_spike = 40.0  # unit: mV
vi_reset = -60.0  # unit: mV
Delta_i = 1.5  # unit: mV
di = 0.0
ai = 0.2
bi = 0.25

# synaptic parameters
g_ampa = 1.0
g_gaba = 1.0
E_ampa = 0.0
E_gaba = -65.0
tau_ampa = 6.0
tau_gaba = 8.0
k_ee = 16.0
k_ei = 16.0
k_ie = 4.0
k_ii = 4.0

# define inputs
T = 4000.0
cutoff = 1000.0
dt = 1e-3
dts = 1e-1
I_ext = np.zeros((int(T/dt), 2))
I_ext[:, 0] += 50.0
I_ext[:, 1] += 36.0
I_ext[int(2000/dt):int(3000/dt), 1] += 14.0
I_ext[int(2500/dt):int(3000/dt), 1] += 25.0

# run the model
###############

N = 10

# initialize model
u_init = np.zeros((N,))
u_init[1] -= 60.0
u_init[6] -= 60.0
model = RNN(1, N, eic_run, C=Ce, C_0=Ci, k=ke, k_0=ki, v_r=ve_r, v_r_0=vi_r, v_t=ve_t, v_t_0=vi_t, v_p=ve_spike,
            v_p_0=vi_spike, v_z=ve_reset, v_z_0=vi_reset, d=de, d_0=di, a=ae, a_0=ai, b=be, b_0=bi, g_ampa=g_ampa,
            g_ampa_0=g_ampa, g_gabaa=g_gaba, g_gabaa_0=g_gaba, E_ampa=E_ampa, E_ampa_0=E_ampa, E_gabaa=E_gaba,
            E_gabaa_0=E_gaba, tau_ampa=tau_ampa, tau_ampa_0=tau_ampa, tau_gabaa=tau_gaba, tau_gabaa_0=tau_gaba,
            weight=k_ee, weight_0=k_ei, weight_1=k_ie, weight_2=k_ii, u_init=u_init, Delta=Delta_e, Delta_0=Delta_i)

# define outputs
outputs = {'se': {'idx': np.asarray([3]), 'avg': False}, 'si': {'idx': np.asarray([4]), 'avg': False}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=I_ext, cutoff=cutoff, fastmath=True)

# plot results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(res["se"]/k_ee)
ax.plot(res["si"]/k_ei)
ax.set_ylabel(r'$s(t)$')
ax.set_xlabel('time')
plt.legend(['RS', 'FS'])
plt.tight_layout()
plt.show()

# save results
pickle.dump({'results': res}, open("results/eic_fre_corrected_het.p", "wb"))
