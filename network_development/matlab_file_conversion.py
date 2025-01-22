from oct2py import octave as oct
from oct2py.utils import Oct2PyError
import multiprocessing as mp
import os

# converting function
def convert_matfile(file: str) -> None:
    try:
        oct.eval(f"load {file}")
        oct.eval(f"save -{target_format} {file}")
    except Oct2PyError as e:
        print(f"The following error occured trying to convert {file} to {target_format}:")
        print(e)

# parameters
path = "/home/richard/data/trujilo_2019"
wdir = os.getcwd()
target_format = "v6"
n_processes = 20

# multiprocessing setup
pool = mp.Pool(processes=n_processes)

# load data and convert it to the target file format
oct.eval(f"cd {path}")
for f in os.listdir(path):
    if f.split(".")[-1] == "mat":
        pool.apply_async(convert_matfile, (f,))

# finish things up
print("Finished the conversion process.")
oct.eval(f"cd {wdir}")
pool.close()
