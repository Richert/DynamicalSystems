import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, mu):
    return mu/(1+np.exp(-x))


mu = 1.0
theta = 0.025
s = 500.0
x = np.linspace(0.0, 0.05, num=1000)
y = sigmoid(s*(x-theta), mu)
plt.plot(x, y)
plt.title("custom sigmoidal function")
plt.show()