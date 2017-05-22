# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
from scipy.spatial.distance import cdist

X1 = np.array([[4, 3],[3, 5],[7, 9]])
X2= np.array([[2, 3], [9, 2], [3, 5]])
X3 = [tuple(a) for a in X1]
X4 = [tuple(a) for a in X2]
print(set(X3) == set(X4))