from Tools.sko.PSO import PSO
import numpy as np
import Enviroments.CCE_ENV_MODIFIED as E

lb = np.zeros(E.TOTAL_TASK_NUM)
ub = np.ones(E.TOTAL_TASK_NUM)
pso = PSO(dim=E.TOTAL_TASK_NUM, pop=40, max_iter=100, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5)
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

record_value = pso.record_value
X_list, V_list = record_value['X'], record_value['V']

fig, ax = plt.subplots(1, 1)
ax.set_title('title', loc='center')
line = ax.plot([], [], 'b.')

# X_grid, Y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 40), np.linspace(-1.0, 1.0, 40))
# Z_grid = demo_func((X_grid, Y_grid))
# ax.contour(X_grid, Y_grid, Z_grid, 20)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

plt.ion()
p = plt.show()


def update_scatter(frame):
    i, j = frame // 10, frame % 10
    ax.set_title('iter = ' + str(i))
    X_tmp = X_list[i] + V_list[i] * j / 10.0
    plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
    return line


ani = FuncAnimation(fig, update_scatter, blit=True, interval=25, frames=300)
plt.show()

ani.save('pso.gif', writer='pillow')

