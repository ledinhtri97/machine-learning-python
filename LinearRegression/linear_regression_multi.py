"""LinearRegression"""
from __future__ import division, print_function, unicode_literals
from sklearn import datasets, linear_model
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

# X should be a matrix with one column
# If X have more feature, add more column
X = np.array([
    [147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183],
    [16, 17, 17, 18, 18, 18, 19, 19, 20, 20, 20, 21, 22]
]).T

# y should be a matrix with one column, same size with X
y = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Show data point
surf = ax.plot_surface(X[:,0], X[:,1], y, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# plt.axis([140, 190, 45, 75])
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (kg)')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# Create matrix same size with X but all elements is 1
one = np.ones((X.shape[0], 1))

# Add to the left side of X
# axis = 1 mean column
# the result is a matrix with only 2 columns
Xbar = np.concatenate((one, X), axis = 1)

# 'dot' mean A*B
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)

# w = A^(-1) *  b
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)

# We know number of features, so we get it directly
# w_0 = w[0][0]
# w_1 = w[1][0]
# x0 = np.linspace(145, 185, 2)
# y0 = w_0 + w_1*x0


# Drawing the fitting line
# plt.plot(X.T, y.T, 'ro')     # data
# plt.plot(x0, y0)               # the fitting line
# plt.axis([140, 190, 45, 75])
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (kg)')
# plt.show()

# fit the model by Linear Regression
# regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
# regr.fit(Xbar, y)

# Compare two results
# print( 'Solution found by scikit-learn  : ', regr.coef_ )
# print( 'Solution found by (5): ', w.T)

