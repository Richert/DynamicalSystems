import matplotlib.pyplot as plt
import pickle

# load data
dataset = "trujilo_2019"
path = "/home/richard-gast/Documents"
save_dir = f"{path}/results/{dataset}"
results = pickle.load(open(f"{save_dir}/{dataset}_prototype_2_fit.pkl", "rb"))

# print fitted parameters
print("Best fit:")
fitted_parameters = {}
for key, val in results["fitted_parameters"].items():
    print(f"{key} = {val}")
    fitted_parameters[key] = val

# plot fit
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(results["target_waveform"], label="target")
ax.plot(results["fitted_waveform"], label="fit")
ax.legend()
ax.set_xlabel("time (ms)")
ax.set_ylabel("norm. fr")
ax.set_title("Fitting results")
plt.show()
