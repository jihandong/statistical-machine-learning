from sklearn.datasets import load_iris
from collections import Counter
import numpy as np
import heapq


class KdTree:
    def __init__(self, x, y, axis=0, dfg=True, debug=False):
        self.axis = axis
        self.left = None
        self.right = None
        self.father = None
        self.cousin = None
        self.distance = None  # Kept for nearest point search.
        self.dfg(x, y) if dfg else self.bfg(x, y)
        if (debug):
            self.display()

    def __lt__(self, other):
        """
        Far point shall be pop as we are searching for nearest neighbors.
        @see KdTree.Neighbors.append(), heapq.pushpop()
        """
        return self.distance > other.distance

    #############################################################################
    # KdTree Generation Methods
    #############################################################################

    def dfg(self, x, y):
        """
        Depth First generation.
        """
        # Choose median ponint.
        sorted_indices = np.argsort(x[:, self.axis])
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
            self.left = KdTree(left_x, left_y, axis)
            self.left.father = self
        if (right_x.size > 0):
            self.right = KdTree(right_x, right_y, axis)
            self.right.father = self

        if (self.left and self.right):
            self.left.cousin = self.right
            self.right.cousin = self.left

    def bfg(self):
        """
        Broad First Generation
        """
        pass

    def display(self, str=""):
        self.id = str
        print("axis = %d, label = %d, median %s = " %
              (self.axis, self.label, str), self.median)
        if (self.left):
            self.left.display(str + "l")
        if (self.right):
            self.right.display(str + "r")

    #############################################################################
    # KdTree Search Methods (knn support)
    #############################################################################

    class Neighbors:
        def __init__(self, k, debug=False):
            self.k = k
            self.debug = debug
            self.neighbors = []

        def append(self, new):
            if len(self.neighbors) < self.k:
                heapq.heappush(self.neighbors, new)
                if self.debug:
                    print(new.median, "(%.3f)" % new.distance, "<=")
            else:
                old = heapq.heappushpop(self.neighbors, new)
                if self.debug and new is not old:
                    print(new.median, "(%.3f)" % new.distance, "<=",
                          old.median, "(%.3f)" % old.distance)

        def farest(self):
            return heapq.nsmallest(1, self.neighbors)[0]
        
        def vote(self):
            labels = [node.label for node in self.neighbors]
            result = Counter(labels)
            return result.most_common(1)[0][0]

    def update(self, x, neighbors):
        """
        Update distance and check neighbors.
        """
        self.distance = np.linalg.norm(self.median - x)
        neighbors.append(self)

    def search(self, x, neighbors, root=None, debug=False):
        """
        Top-down search leaf node.
        """
        node = None
        if x[self.axis] <= self.median[self.axis]:
            node = self.left
        else:
            node = self.right

        # Recursive search branch node.
        if node:
            return node.search(x, neighbors, root)

        # Backtrack after reaching leaf node.
        self.update(x, neighbors)
        self.backtrack(x, neighbors, root)

    def backtrack(self, x, neighbors, root=None):
        """
        Bottom-up check father and cousin nodes.
        """
        if self.father == root:
            return

        # Add father node into neighbors.
        self.father.update(x, neighbors)

        # Check wether cousin node shall be considered.
        if self.cousin:
            x_to_median = abs(x[self.axis] - self.median[self.axis])
            if x_to_median <= neighbors.farest().distance:
                self.cousin.search(x, neighbors, root=self.father)

        # Continue backtrack from father to root.
        self.father.backtrack(x, neighbors, root)

    def vote(self, x, k, debug=False):
        """
        Get k nearest neighbors and vote for classification.
        """
        neighbors = KdTree.Neighbors(k, debug)
        self.search(x, neighbors)
        return neighbors.vote()


class KNN:
    def __init__(self, x, y, p1=0.5, p2=0.25, debug=False):
        self.debug = debug

        # Shuffle the x,y in the same order.
        order = np.random.permutation(x.shape[0])
        x = x[order]
        y = y[order]

        # Split data for training, validation and testing.
        idx1 = int(p1 * x.shape[0])
        idx2 = int((p1 + p2) * x.shape[0])
        self.train_x = x[:idx1]
        self.train_y = y[:idx1]
        self.validate_x = x[idx1:idx2]
        self.validate_y = y[idx1:idx2]
        self.test_x = x[idx2:]
        self.test_y = y[idx2:]
    
    def classify(self, x, y, k):
        """
        Classify base on neighbors' vote.
        """
        classified_y = np.apply_along_axis(lambda p: self.kdtree.vote(p, k),
                                           axis=1, arr=x)
        result = np.sum(classified_y == y)
        return 1.0 * result / x.shape[0]

    def train(self):
        self.kdtree = KdTree(self.train_x, self.train_y, debug=self.debug)

    def validate(self, k1, k2=None):
        # Try parameter from k1 to k2.
        accuracies = [(k, self.classify(self.validate_x, self.validate_y, k))
                      for k in range(k1, (k2 if k2 else k1) + 1)]
        sorted_accuracies = sorted(accuracies, key=lambda p: p[0])
        self.k = sorted_accuracies[0][0]
        print("Validation pick k = %d" % self.k)
        return self.k

    def test(self, k=None):
        accuracy = self.classify(self.test_x, self.test_y, k if k else self.k)
        print("Accuracy = %f" % accuracy)
        return accuracy


if __name__ == '__main__':
    # Load iris data.
    iris = load_iris()
    x = iris.data
    y = iris.target

    # Train and test with KNN model.
    knn = KNN(x, y, debug=True)
    knn.train()
    knn.validate(3, 7)
    knn.test()
