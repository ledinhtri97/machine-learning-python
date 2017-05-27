import numpy as np

Xbar = np.array([])
y = np.array([])


# Define own function
# Example for Linear Regression
def grad(w):
    N = Xbar.shape[0]
    return 1 / N * Xbar.T.dot(Xbar.dot(w) - y)


def cost(w):
    N = Xbar.shape[0]
    return 0.5 * N * np.linalg.norm(y - Xbar.dot(w), 2)**2  # ||y - xbar||**2 & 2


# Caculate grad using bnumeric
def numerical_grad(w,):
    eps = e - 4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps  # f(x + pep)
        w_n[i] -= eps  # f(x- eps)

        # follow CT
        g[i] = (cost(w_p) - cost(w_n))/(2*eps)
    return g


# Check
def check_grad(w):
    # Prepare data to test
    w = np.random.rand(w.shape[0], w.shape[1])

    # Cauclate gradient with two method
    grad1 = grad(w)
    grad2 = numerical_grad(w)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

print( 'Checking gradient...', check_grad(np.random.rand(2, 1)))
