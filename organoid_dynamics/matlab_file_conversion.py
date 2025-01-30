from oct2py import octave as oct
from oct2py.utils import Oct2PyError
import os

# converting function
def convert_matfile(file: str, format: str) -> None:
    try:
        print(f"Trying to convert file {file} to file format {format}...")
        oct.eval(f"load {file}")
        oct.eval(f"save -{format} {file}")
        print("finished conversion successfully.")
    except Oct2PyError as e:
        print(f"The following error occurred trying to convert file {file} to file format {format}:")
        print(e)

# parameters
path = "/home/richard/data/trujilo_2019"
wdir = os.getcwd()
file_format = "v6"

# load data and convert it to the target file format
oct.eval(f"cd {path}")
for f in os.listdir(path):
    if f.split(".")[-1] == "mat":
        convert_matfile(f, file_format)

# finish things up
print("Finished the conversion process.")
oct.eval(f"cd {wdir}")
