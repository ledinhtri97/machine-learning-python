import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# learning rate
alpha = .5

# x0
x = [5.5]

# he so
c_sin = 5

# total of loop
k = 100

# default title
title = '$f(x) = x^2 + %dsin(x)$; ' % c_sin
title += '$x_0 =  %.2f$; ' % x[0]
title += r'$\alpha = %.2f$ ' % alpha

# save filename
file_name = 'gd_14.gif'


# tinh dao ham
def grad(a):
    return 2 * a + c_sin * np.cos(a)


# tinh gia tri
def cost(a):
    return a ** 2 + c_sin * np.sin(a)

# loop k times (maximum)
for it in range(k):
    # generate new x
    # use x[-1] because it is the lastest x
    x_new = x[-1] - alpha * grad(x[-1])

    # khong vuot qua so 1e-3
    if abs(grad(x_new)) < 1e-3:
        break

    # append latest x in the bottom
    x.append(x_new)

# total loop reached
print(it)

# convert to an array
x = np.asarray(x)

# create size of plot
x0 = np.linspace(-4.5, 5.5, 1000)
y0 = cost(x0)

# create plot for x
y = cost(x)
g = grad(x)
plt.plot(x0, y0)
plt.plot(x, y, 'ro', markersize=7)

# create another plot
fig, ax = plt.subplots()


# animation
def update(ii):
    label2 = 'iteration %d/%d: ' % (ii, it) + 'cost = %.2f' % y[ii] + ', grad = %.4f' % g[ii]

    animlist = plt.cla()  # clear prev plot
    # animlist = plt.axis('equal')
    animlist = plt.axis([-6, 6, -8, 30])

    animlist = plt.plot(x0, y0)

    # title += '$\alpha = $ %2f' % alpha
    # animlist = plt.title('$x_0 = $%f, $\alpha = $%f' % (x[0], alpha))
    animlist = plt.title(title)
    if ii == 0:
        animlist = plt.plot(x[ii], y[ii], 'ro', markersize=7)
    else:
        animlist = plt.plot(x[ii - 1], y[ii - 1], 'ko', markersize=7)
        animlist = plt.plot([x[ii - 1], x[ii]], [y[ii - 1], y[ii]], 'k-')
        animlist = plt.plot(x[ii], y[ii], 'ro', markersize=7)

    ax.set_xlabel(label2)
    return animlist, ax

# show animation
anim = FuncAnimation(fig, update, frames=np.arange(0, it), interval=500)
# anim.save(file_name, dpi=100, writer='imagemagick')
plt.show()
