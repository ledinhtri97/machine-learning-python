import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy.random as rnd

# Change default styles
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Set number of data point
# y = 4 + 3x
# Random data point
N = 400
X = 2 * rnd.rand(N, 1)
y = 4 + 3 * X + rnd.randn(N, 1)  # addtion is noise

# Build Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

# create another plot
fig, ax = plt.subplots()


# GD
def grad(w):
    n = Xbar.shape[0]
    return 1/n * Xbar.T.dot(Xbar.dot(w) - y)


def cost(w):
    n = Xbar.shape[0]
    return .5/n*np.linalg.norm(y - Xbar.dot(w), 2)**2


# Caculate grad using bnumeric
def numerical_grad(w, costfc):
    eps = np.e - 4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps  # f(x + pep)
        w_n[i] -= eps  # f(x- eps)

        # follow CT
        g[i] = (costfc(w_p) - costfc(w_n))/(2*eps)
    return g


# Check
def check_grad(w):
    # Prepare data to test
    w = np.random.rand(w.shape[0], w.shape[1])

    # Cauclate gradient with two method
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

print('Checking gradient...', check_grad(np.random.rand(2, 1)))


def gd(w0, grad_fc, eta):
    w = [w0]
    for it in range(100):  # maximum 100 loop
        w_new = w[-1] - eta * grad_fc(w[-1])  # caculate new lost function
        if np.linalg.norm(grad_fc(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
    return w, it

w_init = np.array([[2], [1]]) # create w0
w1, it1 = gd(w_init, grad, 0.5) # eta mean learning rate
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1))


# animation
def update(ii):
    label = "iteration %d/%d " % (ii, it1)

    plt.cla() # clean last plot

    # plot data point
    animlist = plt.plot(X, y, "b.")
    animlist = plt.axis([0, 2, 0, 15])

    # get 2 point to draw a line
    w_0 = w1[ii][0]
    w_1 = w1[ii][1]
    x0 = np.linspace(0, 2, 100)
    y0 = w_0 + w_1 * x0

    # plot line
    animlist = plt.plot(x0, y0, 'r')

    ax.set_xlabel(label)
    return animlist, ax

anim = FuncAnimation(fig, update, frames=np.arange(0, it1), interval=500)
plt.show()

