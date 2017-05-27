import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from KmeansApplication.display_network import display_network

# Load MNIST lib
mndata = MNIST('C:/Users/DELL/PycharmProjects/MachineLearning/MNIST/')

# Only using test data
mndata.load_testing()
X = mndata.test_images

# Convert input to an array
# with all element (0;1)
X0 = np.asarray(X)[:1000, :] / 256.0
X = X0

# Set num of cluster
K = 10

# Trainning data
kmeans = KMeans(n_clusters=K).fit(X)

# Use that data to predict
pred_label = kmeans.predict(X)

print(kmeans.cluster_centers_.T.shape)

# Display center of cluster with table
A = display_network(kmeans.cluster_centers_.T, K, 1)

# Display image
f1 = plt.imshow(A, interpolation='nearest', cmap="jet")
f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
plt.show()

# Save image
# plt.savefig('a1.png', bbox_inches='tight')

# a colormap and a normalization instance
cmap = plt.cm.jet
norm = plt.Normalize(vmin=A.min(), vmax=A.max())

# map the normalized data to colors
# image is now RGBA (512x512x4)
image = cmap(norm(A))

# Save image with RGBA
# import scipy.misc
# scipy.misc.imsave('aa.png', image)

print(pred_label.shape)

# chose 20 sample point to display
N0 = 20
X1 = np.zeros((N0 * K, 784))
X2 = np.zeros((N0 * K, 784))

# With each cluster, get random sample data
for k in range(K):
    # get all data that have predict label = k
    Xk = X0[pred_label == k, :]

    # trainning that data and fit it to cluster
    center_k = [kmeans.cluster_centers_[k]]
    neigh = NearestNeighbors(N0).fit(Xk)

    # With that center, find N0 neighbors that closest to the center
    # Caculate distance and get the id of them
    dist, nearest_id = neigh.kneighbors(center_k, N0)

    # Set element of X1 with row from N0*k to N0*k + N0 with the id
    X1[N0 * k: N0 * k + N0, :] = Xk[nearest_id, :]

    # Set element of X2 with row from N0*k to N0*k + N0 with element
    X2[N0 * k: N0 * k + N0, :] = Xk[:N0, :]

# show sample
plt.axis('off')
A = display_network(X2.T, K, N0)
f2 = plt.imshow(A, interpolation='nearest')
plt.gray()
plt.show()
