import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from matplotlib.colors import to_hex

# preparations
##############

# plot settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (4.5, 2.5)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 8.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['axes.titlepad'] = 1.0
labelpad = 1.0
plt.rcParams['axes.labelpad'] = labelpad
markersize = 15
cmap = sns.color_palette("plasma", as_cmap=False, n_colors=4)


# QIF evolution function
def qif_rhs(v, tau, I):
    return (v**2 + I)/tau


# calculations
##############

# calculate phase space curves for different values of I
n1 = 1000
tau = 1.0
Is = [-1.0, 0.0, 1.0]
vs = np.linspace(-2, 2, n1)
dvs = [qif_rhs(vs, tau, inp) for inp in Is]

# calculate f-I curve of qif neuron
n2 = 1000
v_r = -100.0
v_t = 100.0
I = np.linspace(-10, 10, n2)
s = np.zeros_like(I)
I_sqrt = np.sqrt(I[I >= 0])
s[I >= 0] = I_sqrt/(tau*(np.arctan(v_t/I_sqrt) - np.arctan(v_r/I_sqrt)))

# calculate qif solution for time-varying input
T = 40.0
dt = 0.01
n_steps = int(T/dt)
time = np.linspace(0, T, n_steps)
I_base = -1.5
inp = np.zeros_like(time)
in_durs = [200, 200, 200, 1000, 1000]
in_starts = [500, 1000, 1500, 2000, 3000]
in_strengths = [1.0, 2.0, 3.0, 2.0, 3.0]
for dur, start, amp in zip(in_durs, in_starts, in_strengths):
    inp[start:start+dur] = amp

v_col = []
v = -2.0
for step in range(n_steps):
    v += dt*qif_rhs(v, tau, I_base+inp[step])
    if v > v_t:
        v = v_r
    v_col.append(v)


# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=3, figure=fig)

# plotting of v-dv curves
all_dvs = np.asarray(dvs).flatten()
y_min, y_max = np.min(all_dvs), np.max(all_dvs)
for i, (I_tmp, dv) in enumerate(zip(Is, dvs)):
    ax = fig.add_subplot(grid[0, i])
    ax.plot(vs, dv, color=to_hex(cmap[0]))
    ax.plot(vs, np.zeros_like(vs), color=to_hex(cmap[-1]))
    ax.set_xlabel(r'$V_i$')
    ax.set_ylabel(r'$\dot V_i$')
    ax.set_title(rf'$I = {int(I_tmp)}$')
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-2.0, 2.0)

# dummy for theta neuron plot
ax = fig.add_subplot(grid[1, 0])
ax.set_xlabel(r'$\alpha = 0$, $\alpha = 0.0125$')
ax.set_ylabel(r'$\alpha = 0.025$, $\alpha = 0.05$')
ax.set_title(r'$\alpha = 0.1$, $\alpha = 0.2$')

# f-I curve
ax = fig.add_subplot(grid[1, 1])
ax.plot(I, s, color=to_hex(cmap[0]))
ax.set_xlabel(r'$I$')
ax.set_ylabel(r'$s_i$ in $\frac{1}{\tau}$')

# qif dynamics
ax = fig.add_subplot(grid[1, 2])
ax.plot(time, np.asarray(v_col),  color=to_hex(cmap[0]))
ax.set_xlabel(r't in units of $\tau$')
ax.set_ylabel(r'$V_i$')
ax.set_ylim(-20, 20)
ax.set_xticks([0.0, 20.0, 40.0])
ax2 = ax.twinx()
I_total = np.squeeze(I_base+inp)
ax2.plot(time, I_total, color=to_hex(cmap[-1]))
ax2.set_ylabel(r'$I(t)$')
ax2.set_ylim(-2.0, 6.0)
ax2.set_yticks([-2.0, 2.0, 6.0])

# padding
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.01, hspace=0., wspace=0.)

# saving
fig.canvas.draw()
plt.savefig(f'qif_neuron.svg')
plt.show()
