import pickle

condition = "iSPN_control"
fn = f"/home/rgf3807/PycharmProjects/DynamicalSystems/BasalGanglia/STR_EPSPs/{condition}.pkl"
save_to_file = ["parameters", "fitted_epsps", "target_epsps"]
data = pickle.load(open(fn, "rb"))
for s in save_to_file:
    data[s].to_csv(f"{condition}_{s}.csv")
    #data[s].to_excel(f"{condition}_{s}.xlsx")
