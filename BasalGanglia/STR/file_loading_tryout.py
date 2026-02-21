from scipy.io import loadmat
import os

# specify paths and conditions
mount_dir = "/mnt/kennedy_lab_data"
condition = "amph"
drug = "haloperidol"
dose = "LowDose"
mouse = "m971"

# locate the correct file to load
path = f"{mount_dir}/Parkerlab/neural_data/{drug}/{dose}"
for file in os.listdir(path):
	if mouse in file:
		if condition == "amph" and condition in file:
			break
		elif condition == "veh" and condition not in file:
			break
else:
	raise FileNotFoundError("File {file} not found at {path}.")

# load the data
data = loadmat(f"{path}/{file}/{condition}_drug.mat", simplify_cells=True)
print(list(data.keys()))
print(list(data[f"{condition}_drug"].keys()))