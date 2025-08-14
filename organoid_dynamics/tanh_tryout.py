import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, mu):
    return mu/(1+np.exp(-x))


mu = 1.0
theta = 20.0
s = 0.2
x = np.linspace(0.0, 40.0, num=1000)
y = sigmoid(s*(theta-x), mu)
plt.plot(x, y)
plt.title("custom sigmoidal function")
plt.show()