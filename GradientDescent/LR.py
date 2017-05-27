import numpy as np
import numpy.random as rnd
import os
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
rnd.seed(42)

# To plot pretty figures
# matplotlib inline
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

N = 400
X = 2 * rnd.rand(N, 1)
y = 4 + 3 * X + rnd.randn(N, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()

