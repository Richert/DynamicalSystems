from oct2py import octave as oct
import os

path = "/media/richard/data/trujilo_2019"
wdir = os.getcwd()

oct.eval(f"cd {path}")
for f in os.listdir(path):
    if f.split(".")[-1] == "mat":
        oct.eval(f"save -v7 {f}")
oct.eval(f"cd {wdir}")
