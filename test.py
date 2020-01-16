import matplotlib.pyplot as plt
import numpy as np


data = np.load("./results/data_0.npy")
ddpg = np.load("./results/data_1.npy")

td = plt.plot(data, c="r")
d = plt.plot(ddpg, c="b")

plt.show()