import matplotlib.pyplot as plt
from pyauto import PyAuto
import sys
sys.path.append('../../')


################
# data loading #
################

# load pyauto instance from file
fname = 'QIF_population/results/qif_sfa_syns.pkl'
a = PyAuto.from_file(fname, auto_dir="~/PycharmProjects/auto-07p")

#################
# visualization #
#################

fig, axes = plt.subplots(nrows=2, figsize=(12, 8))

# plot steady-state solutions for multiple values of alpha
##########################################################

ax = axes[0]
etas = a.additional_attributes['etas']

# steady-state solutions for multiple values of alpha
for i in range(len(etas)):
    ax = a.plot_continuation('PAR(2)', 'U(1)', cont=f'J_{i}', ax=ax)
ax.set_xlabel('J')
ax.set_ylabel('r')
plt.legend([fr'$\eta = {a}$' for a in etas])

# plot bifurcation diagram for particular value of alpha
########################################################

ax = axes[1]
idx = 2

# steady-state solutions
ax = a.plot_continuation('PAR(2)', 'U(1)', cont=f'J_{idx}', ax=ax)

# limit cycle solutions
try:
    ax = a.plot_continuation('PAR(2)', 'U(1)', cont=f'J_hb1', ax=ax, ignore=['BP'])
except KeyError:
    pass
try:
    ax = a.plot_continuation('PAR(2)', 'U(1)', cont=f'J_hb2', ax=ax, ignore=['BP'])
except KeyError:
    pass

ax.set_xlabel('J')
ax.set_ylabel('r')
ax.set_title(rf'$\eta = {etas[idx]}$')

plt.tight_layout()
plt.show()
