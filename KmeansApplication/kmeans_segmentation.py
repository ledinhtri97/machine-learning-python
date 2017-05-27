import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Load image
img = mpimg.imread('baithi.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()

# Img have [y, x, z] with z = 3
# y = height, x= width
# Convert image to 1 matrix with N pixel with 3 column for Red, Green, Blue
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

# K = 3 cluster
for K in [2]:
    kmeans = KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)

    img4 = np.zeros_like(X)
    # replace each pixel by its center
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k]
    # reshape and display output image
    img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img5, interpolation='nearest')
    plt.axis('off')
    plt.show()