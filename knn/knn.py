from sklearn.datasets import load_iris
import numpy as np


class KdTree:
    def __init__(self, x, y, axis=0, df=True):
        self.axis = axis
        self.left = None
        self.right = None
        self.father = None
        self.distance = None # Kept for nearest point search.
        if df:
            self.dfg(x, y) # Depth first.
        else:
            self.bfg(x, y) # Broad first.

    def dfg(self, x, y):
        # Choose median ponint.
        sorted_indices = np.argsort(x[:,self.axis])
        sorted_x = x[sorted_indices]
        sorted_y = y[sorted_indices]
        median_idx = sorted_x.shape[0] // 2

        # Split data into left, median, and right parts.
        left_x = sorted_x[:median_idx, :]
        left_y = sorted_y[:median_idx]
        right_x = sorted_x[median_idx + 1:, :]
        right_y = sorted_y[median_idx + 1:]
        self.median = sorted_x[median_idx]
        self.label = sorted_y[median_idx]

        # Recursively generate the subtrees.
        axis = (self.axis + 1) % left_x.shape[1]
        if (left_x.size > 0):
            self.left = KdTree(left_x, left_y, axis, df=True)
            self.left.father = self
        if (right_x.size > 0):
            self.right = KdTree(right_x, right_y, axis, df=True)
            self.right.father = self

    def bfg(self):
        pass

    def show(self, str=""):
        print("axis = %d, label = %d, median %s = " %\
              (self.axis, self.label, str), self.median)
        if (self.left):
            self.left.show(str + "l")
        if (self.right):
            self.right.show(str + "r")
        pass

    def distance(self, x1, x2):
        pass

    def search(self, x):
        """
        Top-down search leaf node.
        """
        node = self.left if x[self.axis] <= self.median[self.axis]\
            else self.right

        return node.search() if node else self

    def backtrack(self, x, k, points, distances):
        """
        Bottom-up check father and cousin nodes.
        """
        pass
        


class KnnClassifier:
    def __init__(self, x, y, k):
        self.kdtree = KdTree(x, y)
        self.kdtree.show()
        self.k = k

    def classify(self):
        pass

    def train(self):
        pass

    def validate(self):
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

