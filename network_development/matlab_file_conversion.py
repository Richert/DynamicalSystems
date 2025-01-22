from oct2py import octave as oct
import os

path = "/home/richard/data/trujilo_2019"
wdir = os.getcwd()

# load data and convert it to a file format that can be read via scipy.io.loadmat
oct.eval(f"cd {path}")
for f in os.listdir(path):
    if f.split(".")[-1] == "mat":
        oct.eval(f"load {f}")
        oct.eval(f"save -v7 {f}")
oct.eval(f"cd {wdir}")
