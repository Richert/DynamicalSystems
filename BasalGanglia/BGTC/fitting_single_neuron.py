import numpy as np


# function definitions
######################

def ik(y, I_ext, s, C, k, v_r, v_t, a, b, d):

    v = y[0]
    u = y[1]

    dv = (k*(v-v_r)*(v-v_t) + I_ext - u) / C
    du = a*(b*(v-v_r) - u) + d*s

    return np.asarray([dv, du])


def hh(y, I_ext, )
    pass
