from __future__ import division, print_function, unicode_literals
# Import libaries
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import matplotlib.animation as animation
import random
from sklearn.cluster import KMeans as km

# Set seed to random
# np.random.seed(11)

# Create E, N
means = [[2, 2], [8, 3], [3, 6]]
conv = [[1, 0], [0, 1]]

# Number of data for each cluster
N = 500

# Ramdom to create data point
X0 = np.random.multivariate_normal(means[0], conv, N)
X1 = np.random.multivariate_normal(means[1], conv, N)
X2 = np.random.multivariate_normal(means[2], conv, N)

X = np.concatenate((X0, X1, X2), axis = 0)  # Combie data
K = 3 # Number of cluster

# Function that use to plot
def kmeans_display(X, label):
    K = np.amax(label) + 1  # label start from 0, so it should be add one

    # Find all element that have label = 0, after that get label = 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    # First col mean x, and second col mean y
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()

# Function that random center
def kmeans_init_centers(X, k):
    # func np.random.choice random k number base on X.shape[0]
    # X[...] will get point at that position
    return X[np.random.choice(X.shape[0], k, replace=False)]

# Sign label to data by caculate distance from X to center
def kmeans_assign_labels(X, centers):
    # Caculate distance
    # Each col for each cluster, each data for each distance
    D = cdist(X, centers)

    # return 0 and 1
    # 1 show that point in that label
    return np.argmin(D, axis=1)

# Update the label to the point
def kmeans_update_centers(X, labels, K):
    # Create the matrix with K row and X.shape[1] col
    centers = np.zeros((K, X.shape[1]))
    # Loop all cluster
    for k in range(K):
        # With each cluster, get all X with that have label = k
        # for example
        # label = [1 0 0 2 1 2 1 1 1]
        # we need k = 0, so we take row 2, 3
        # we need k = 1, so we take row 1, 5, 7, 8 , 9
        # we need k = 2, so we take row 4, 6
        Xk = X[labels == k, :]

        # Update center k th (all col) with average follow row to row (vertical)
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

# Check that center have change or not
def has_converged(centers, new_centers):
    # Create a set contain all center with order
    # So if 2 set that equal is have a same set with same order
    # else they have difference order
    # If all element in center same as all element in new_center, that ook
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

# main function
def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]   # create the list of center
    labels = [] # create empty labels
    it = 0  # count loop
    while True:
        # centers[-1] mean get from bottom, the new ones (the new center)
        # kmeans_assign_labels return array with 0 and 1
        # labels after that is [1, 1, 1, 2, 2, 2, 0, 0, 0] for the last center
        labels.append(kmeans_assign_labels(X, centers[-1]))

        # We have X and it label, so we update new centers with lasest labels
        new_centers = kmeans_update_centers(X, labels[-1], K)

        # We check two centers
        if has_converged(centers[-1], new_centers):
            break

        # with new center, we append it to the centers
        # so that we use centers[-1] to get the new ones
        # we can't use '==' because centers is the list with many centers
        centers.append(new_centers)

        it += 1 # increase
    return (centers, labels, it)

# Create label, 0 for data 1, 1 for data 2,...
original_label = np.asarray([0]*N + [1]*N + [2]*N).T

# Display
kmeans_display(X, original_label)

(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])

kmeans_display(X, labels[-1])

# kmeans2 = km(n_clusters=3, random_state=0).fit(X)
# print('Centers found by scikit-learn:')
# print(kmeans2.cluster_centers_)
# pred_label = kmeans2.predict(X)
# kmeans_display(X, pred_label)

###########
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(centers[-1])

def voronoi_finite_polygons_2d(vor, radius=None):

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

fig, ax = plt.subplots()

def update(ii):
    label2 = 'iteration {0}: '.format(ii/2)
    if ii%2:
        label2 += ' update centers'
    else:
        label2 += ' assign points to clusters'

    i_c = int((ii+1)/2)
    i_p = int(ii/2)

    label = labels[i_p]
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    animlist = plt.cla()
    animlist = plt.axis('equal')
    animlist = plt.axis([-2, 12, -3, 12])

    animlist = plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    animlist = plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    animlist = plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    # display centers and voronoi
    i = i_c
    animlist = plt.plot(centers[i][0, 0], centers[i][0, 1], 'y^', markersize = 15)
    animlist = plt.plot(centers[i][1, 0], centers[i][1, 1], 'yo', markersize = 15)
    animlist = plt.plot(centers[i][2, 0], centers[i][2, 1], 'ys', markersize = 15)

    ## vonoroi
    points = centers[i]
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius = 1000)
    for region in regions:
        polygon = vertices[region]
        animlist = plt.fill(*zip(*polygon), alpha=.2)


    ax.set_xlabel(label2)
    return animlist, ax

anim = FuncAnimation(fig, update, frames=np.arange(0, 2*it), interval=1000)
# anim.save('kmeans.gif', dpi=200, writer='imagemagick')
plt.show()

