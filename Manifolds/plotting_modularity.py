import pickle
import numpy as np
from pyauto import PyAuto
from pyrecu import modularity
import matplotlib.pyplot as plt
import sys
cond = 0 #sys.argv[-1]

# import data
#############

# load mean-field data
a = PyAuto.from_file("results/ik_bifs.pkl")

# load rnn data
data = pickle.load(open(f"results/rnn_{cond}.p", "rb"))
res = data['results']
etas = data['etas']
p = data['p']
W = data['W']

# calculate