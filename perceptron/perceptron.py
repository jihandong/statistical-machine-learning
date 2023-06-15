from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


class Perceptron:
    def __init__(self, x, y, p, stride):
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

        # Configure with hyper-parameters.
        self.configure(np.random.rand(self.train_x.shape[1]), 0, stride)

    def configure(self, w, b, stride):
        self.w = w
        self.b = b
        self.stride = stride

    def perceive(self, x, y):
        # Apply classify function.
        predict_y = np.dot(x, self.w) + self.b
        result = y * predict_y

        # Get mis-classify entries.
        error_entries = np.where(result <= 0)[0]
        return error_entries

    def train(self):
        start_time = time.perf_counter()
        round = 0

        # Stochastic Gradient Descent
        while (1):
            round = round + 1
            error_entries = self.perceive(self.train_x, self.train_y)
            if (error_entries.size > 0):
                idx = np.random.choice(error_entries)
                # Update paramters.
                self.w = self.w + self.stride * \
                    self.train_y[idx] * self.train_x[idx]
                self.b = self.b + self.stride * self.train_y[idx]
            else:
                break

        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Training takes %dr, costs %fs" % (round, run_time))

    def test(self):
        error_entries = self.perceive(self.test_x, self.test_y)
        accuracy = 1.0 - 1.0 * error_entries.size / self.test_x.shape[0]
        print("Testing accuracy = %f" % accuracy)

    def show(self):
        if (self.x.shape[1] > 2):
            return

        # Random choose two features to display.
        idx1, idx2 = np.random.choice(self.x.shape[1], 2, replace=False)

        # Draw hyper-space.
        left = self.x[:, idx1].min()
        right = self.x[:, idx1].max()
        length = 0.15 * (right - left)
        line_x = np.linspace(left + length, right - length)
        line_y = - (self.w[idx1] * line_x + self.b) / self.w[idx2]
        plt.plot(line_x, line_y)

        # Draw samples.
        xa = self.x[self.y == 1]
        xb = self.x[self.y == -1]
        plt.scatter(xa[:, idx1], xa[:, idx2], label="Positive")
        plt.scatter(xb[:, idx1], xb[:, idx2], label="Negative")

        plt.legend()
        plt.show()


if __name__ == '__main__':
    # Prepare iris data.
    iris = load_iris()
    x = iris.data
    y = np.where(iris.target > 0, 1, -1)

    # Apply classic perceptron model.
    perceptron = Perceptron(x, y, 0.7, 0.1)
    perceptron.train()
    perceptron.test()
    perceptron.show()
