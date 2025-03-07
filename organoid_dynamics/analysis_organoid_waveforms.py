import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in data
data = pd.read_csv("/home/richard-gast/Documents/data/trujilo_2019/trujilo_2019_waveforms.csv")

print(data.summary)
