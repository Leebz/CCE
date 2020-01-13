import matplotlib.pyplot as plt
import numpy as np






data = np.random.random(10)
print(data)

# plt.scatter(np.arange(len(data)), data)
plt.plot(data, c = 'r')
plt.show()