from sklearn.datasets import load_iris
import numpy as np
import time

class NaiveBayes:
    def __init__(self, x, y, p=0.7):
        # Shuffle the x,y in the same order.
        order = np.random.permutation(x.shape[0])
        self.x = x[order]
        self.y = y[order]

        # Split data for training and testing.
        idx = int(p * x.shape[0])
        self.train_x = self.x[:idx]
        self.train_y = self.y[:idx]
        self.test_x = self.x[idx:]
        self.test_y = self.y[idx:]
        print("Load %d samples, %d for traing, %d for testing"
              % (x.shape[0], idx, x.shape[0] - idx))
        
        # 

    def calculate(self):
        """
        Calculate probabilities.
        """
        # Acknowledge classes.
        classes = np.unique(self.train_y)
        nb_classes = len(classes)
        self.p_y = np.zeros(nb_classes)

        # Acknowledge features.
        # FIXME: need to tolerate some floating-point comparison errors.
        nb_samples, nb_features = self.train_x.shape
        ranges_x = [np.unique(self.train_x[:,j]) for j in range(nb_features)]

        self.p_y = np.zeros(nb_classes)
        # TODO: self.p_x_y = [None] * 
        for i, c in enumerate(classes):
            # Calculate P(Y = y).
            x_with_yc = self.train_x[self.train_y == c]
            self.p_y[i] = 1.0 * x_with_yc.shape[0] / nb_samples

        # Calculate P(Xj|Y)



if __name__ == '__main__':
    # Prepare iris data.
    iris = load_iris()
    x = iris.data
    y = iris.target

    # Apply naive bayes.
    model = NaiveBayes(x, y)
    model.calculate()