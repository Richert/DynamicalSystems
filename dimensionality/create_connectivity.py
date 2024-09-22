from rectipy import random_connectivity
import pickle

import sys
from custom_functions import *

N = 1000
p = 0.2
W = random_connectivity(N, N, p, normalize=True)

pickle.dump({"W": W}, open("config/spn_connectivity.pkl", "wb"))
