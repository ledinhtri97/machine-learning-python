import numpy as np
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
iris = datasets.load_iris()
iris_X = iris.data  # data
iris_y = iris.target  # label

# Print number of classes and label
print("Number of classes: %d" % len(np.unique(iris_y)))
print("Number of data points: %d" % len(iris_y))

# chose some sample to see some special attr
for i in range(0, 3):
    X0 = iris_X[iris_y == i, :]
    print("Sample from class ", i, ":\n", X0[:5, :])

# split data to 2 sets
# 50 for test data
# 100 for trainning data
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=50)

# Check size
print("Training size: %d" % len(y_train))
print("Test size: %d" % len(y_test))

# Using 10 neighbors to caculate
# p = 2 min norm 2 that is the distance between two point
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Result
print("Print results for 20 test data points:")
print("Predicted labels: ", y_pred)
print("Ground truth    : ", y_test)

print("Accuracy: %.2f %%" % (100 * accuracy_score(y_test, y_pred)))


# weights
# 'distance' to vote nearest distance higher score
# 'uniform' to vote all point as same
# my_weights to customize
def my_weights(distance):
    sigma2 = 5
    return jknp.exp(-distance ** 2 / sigma2)
