import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return 0.5 + 0.5*np.tanh(1.0*(x - 10.0))


x = np.linspace(0.0, 40.0, num=1000)
y = tanh(x)
plt.plot(x, y)
plt.title("custom tanh function")
plt.show()