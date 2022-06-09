import numpy as np
from pyrecu import sequentiality
import pickle

data = pickle.load(open("results/spn_rnn.p", "rb"))['results']

seq = sequentiality(data['s'].T, mode='same')
print(rf'$s = {seq}$')
