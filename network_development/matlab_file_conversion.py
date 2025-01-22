from oct2py import octave as oct
from oct2py.utils import Oct2PyError
import os

path = "/home/richard/data/trujilo_2019"
wdir = os.getcwd()
target_format = "hdf5"

# load data and convert it to the target file format
oct.eval(f"cd {path}")
for f in os.listdir(path):
    if f.split(".")[-1] == "mat":
        try:
            print(f"Converting file {f} to {target_format}...")
            oct.eval(f"load {f}")
            oct.eval(f"save -{target_format} {f}")
        except Oct2PyError as e:
            print(f"The following error occured trying to convert {f}:")
            print(e)
            pass
print("Finished the conversion process.")
oct.eval(f"cd {wdir}")
