import numpy as np
from mnist import MNIST
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score
import time

# Load data
mndata = MNIST('C:/Users/DELL/PycharmProjects/MachineLearning/MNIST/')
mndata.load_training()
mndata.load_testing()
X_test = mndata.test_images
X_train = mndata.train_images
y_test = np.asarray(mndata.test_labels)
y_train = np.asarray(mndata.train_labels)

# Set time to caculate time execution
start_time = time.time()

# Use KNN
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p= 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Caculate the end time
end_time = time.time()

# Result
print("Accuracy of 1NN for MNIST: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
print("Running time: %.2f (s)" % (end_time - start_time))
