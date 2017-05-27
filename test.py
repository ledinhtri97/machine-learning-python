import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

mndata = MNIST('C:/Users/DELL/PycharmProjects/MachineLearning/MNIST/')

mndata.load_testing()
X = mndata.test_images

X0 = np.asarray(X)[:1000, :] / 256.0

