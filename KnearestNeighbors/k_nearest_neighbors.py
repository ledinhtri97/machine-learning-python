import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print("Number of classes: %d" %len(np.unique(iris_y)))
print("Number of data points: %d" %len(iris_y))

for i in range(0,3):
    X0 = iris_X[iris_y == i,:]
    print("Sample from class ", i, ":\n", X0[:5,:])

X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size= 50)

print("Training size: %d" %len(y_train))
print("Test size: %d" %len(y_test))

clf = neighbors.KNeighborsClassifier(n_neighbors= 10, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Print results for 20 test data points:")
print("Predicted labels: ", y_pred)
print("Ground truth    : ", y_test)

print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
