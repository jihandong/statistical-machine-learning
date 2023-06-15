from sklearn.datasets import load_iris
import numpy as np


class KdTree:
    def __init__(self, data, axis=0, df=True):
        self.axis = axis
        self.left = None
        self.right = None
        self.median = None
        if df:
            self.dfg(data) # Depth first.
        else:
            self.bfg(data) # Broad first.

    def dfg(self, data):
        # Choose median ponint.
        sorted_indices = np.argsort(data[:,self.axis])
        sorted_data = data[sorted_indices]
        median_idx = sorted_data.shape[0] // 2

        # Split data into left, median, and right parts.
        left = sorted_data[:median_idx, :]
        right = sorted_data[median_idx + 1:, :]
        self.median = data[median_idx]

        # Recursively generate the subtrees.
        axis = (self.axis + 1) % data.shape[1]
        if (left.size > 0):
            self.left = KdTree(left, axis, df=True)
        if (right.size > 0):
            self.right = KdTree(right, axis, df=True)

    def bfg(self):
        pass

    def show(self, str=""):
        print("axis =", self.axis, "median", str, "=", self.median)
        if (self.left):
            self.left.show(str + "l")
        if (self.right):
            self.right.show(str + "r")
        pass


class KnnClassifier:
    def __init__(self, x, y, k):
        self.kdtree = KdTree(x)
        self.kdtree.show()
        self.k = k

    def classify(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


if __name__ == '__main__':
    # Load iris data.
    iris = load_iris()
    x = iris.data
    y = iris.target

    # Train and test with KNN model.
    knn = KnnClassifier(x, y, 3)

