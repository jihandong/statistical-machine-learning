import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

class Perceptron:
    def __init__(self):
        self.__load(0.7) # percentage of training data.
        self.stride = 0.1 # learning rate.
        self.w = np.random.rand(self.train_x.shape[1], dtype=np.float32)
        self.b = 0

    def __load(self, p):
        # 1. Prepare data for perceptron.
        iris = load_iris()
        x = iris.data
        y = iris.target
        y = np.where(iris.target > 0, 1, -1)

        # 2. Shuffle the x,y in the same order.
        order = np.random.permutation(x.shape[0])
        x = x[order]
        y = y[order]

        # 3. Split for training and testing.
        idx = int(p * x.shape[0])
        self.train_x = x[:idx]
        self.train_y = y[:idx]
        self.test_x = x[idx:]
        self.test_y = y[idx:]

    def __predict(self):
        predict_y = np.dot(self.w, self.train_x) + self.b
        predict_y = np.where(predict_y > 0, 1, -1)
        result = self.train_y * predict_y
        return result

    def __update(self, idx):
        w = self.w
        b = self.b
        self.w = w + self.stride * self.train_y[idx] * self.train_x[idx]
        self.b = b + self.stride * self.train_y[idx]

    def train(self):
        while (1):
            # 1. Apply w,b and pick an mis-classified sample.
            result = self.__predict()
            error_result = np.where(result < 0)[0]
            if (error_result.size > 0):
                index = np.random.choice(error_result)
                self.__update(index)
            else:
                break
        print()

    def test(self):
        pass