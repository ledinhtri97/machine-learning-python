import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


# Define cost fundtion
def cost(w):
    x = w[0]
    y = w[1]
    return (x ** 2 + y - 7) ** 2 + (x - y + 1) ** 2


# Defin grad function
def grad(w):
    x = w[0]
    y = w[1]
    g = np.zeros_like(w)
    g[0] = 2 * (x ** 2 + y - 7) * 2 * x + 2 * (x - y + 1)
    g[1] = 2 * (x ** 2 + y - 7) + 2 * (y - x - 1)
    return g


# Caculate grad using numerical method
def numerical_grad(w, cost):
    eps = 1e-6
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n)) / (2 * eps)
    return g


# Compare numerical method with formular
def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-4 else False


# Check result
w = np.random.randn(2, 1)
# w_init = np.random.randn(2, 1)
print('Checking gradient...', check_grad(w, cost, grad))


# GD
def gd(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta * grad(w[-1])
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
            break
        w.append(w_new)
    # print('iter %d: ' % it, w[-1].T)
    return w, it


# Define a eta
eta = 0.016

# Get GD result
w_init = np.array([[-5], [-5]])
# w_init = np.random.randn(2, 1)
w1, it1 = gd(w_init, grad, eta)
print(w1[-1])

# Create point
delta = 0.025
x = np.arange(-6.0, 5.0, delta)
y = np.arange(-20.0, 15.0, delta)
X, Y = np.meshgrid(x, y)
Z = (X ** 2 + Y - 7) ** 2 + (X - Y + 1) ** 2

# Create w0
# Caculate w using gd
w_init = np.array([[-5], [-5]])
w, it = gd(w_init, grad, eta)

# Another w0
w_init = np.array([[0], [6]])
w2, it = gd(w_init, grad, eta)

# Create plot view
fig, ax = plt.subplots(figsize=(8, 5))
plt.cla()
plt.axis([1.5, 6, 0.5, 4.5])
#     x0 = np.linspace(0, 1, 2, endpoint=True)
title = '$f(x, y) = (x^2 + y -7)^2 + (x - y + 1)^2$'


# animation
def update(ii):
    if ii == 0:
        plt.cla()

        CS = plt.contour(X, Y, Z, np.concatenate((np.arange(0.1, 50, 5), np.arange(60, 200, 10))))
        manual_locations = [(-4, 15), (-2, 0), (1, .25)]
        animlist = plt.clabel(CS, inline=.1, fontsize=10, manual=manual_locations)
        animlist = plt.title('$f(x, y) = (x^2 + y -7)^2 + (x - y + 1)^2$')
        plt.plot([-3, 2], [-2, 3], 'go')
    else:
        animlist = plt.plot([w[ii - 1][0], w[ii][0]], [w[ii - 1][1], w[ii][1]], 'r-')
        animlist = plt.plot([w2[ii - 1][0], w2[ii][0]], [w2[ii - 1][1], w2[ii][1]], 'r-')

    # Connect 2 point with a line
    animlist = plt.plot(w[ii][0], w[ii][1], 'ro')
    animlist = plt.plot(w2[ii][0], w2[ii][1], 'ro')

    # Retext label
    xlabel = '$\eta =$ ' + str(eta) + '; iter = %d/%d' % (ii, it)

    ax.set_xlabel(xlabel)
    return animlist, ax


anim1 = FuncAnimation(fig, update, frames=np.arange(0, it), interval=200)
plt.show()
