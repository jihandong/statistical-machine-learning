from sklearn.datasets import load_iris
import numpy as np
import time

class NaiveBayes:
    def __init__(self, x, y, p=0.7, smooth=1, debug=False):
        # Configure with hyper-parameters.
        self.p = p
        self.smooth = smooth
        self.debug = debug
        self.delim = "-" * 64 + "\n"

        # Shuffle the x,y in the same order.
        order = np.random.permutation(x.shape[0])
        self.x = x[order]
        self.y = y[order]

        # Split data for training and testing.
        idx = int(self.p * x.shape[0])
        self.train_x = self.x[:idx]
        self.train_y = self.y[:idx]
        self.test_x = self.x[idx:]
        self.test_y = self.y[idx:]
        if (self.debug):
            print(self.delim, "Load %d samples, %d for traing, %d for testing"
                % (x.shape[0], idx, x.shape[0] - idx))

    def train(self):
        """
        Calculate probabilities.
        """
        start_time = time.perf_counter()

        # Acknowledge classes.
        self.classes = np.unique(self.train_y)
        nb_classes = len(self.classes)

        # Acknowledge features.
        # FIXME: need to tolerate some floating-point comparison errors.
        nb_samples, nb_features = self.train_x.shape
        self.features = [np.unique(self.train_x[:,j])
                         for j in range(nb_features)]

        # Calulate probabilities.
        # TODO: highly possible (but hard) to optimize, especially list.
        self.p_y = np.zeros(nb_classes)
        self.p_x_y = []
        for i, class_ in enumerate(self.classes):
            # Calculate P(Y = y).
            x_with_yc = self.train_x[self.train_y == class_]
            nb_samples_yc = x_with_yc.shape[0]
            self.p_y[i] = 1.0 * nb_samples_yc / nb_samples
            self.p_x_y.append([])

            for j, feature in enumerate(self.features):
                # Calculate P(Xj|Y = y).
                range_feature_xj = feature.size
                self.p_x_y[i].append(np.zeros(range_feature_xj))

                for k, value in enumerate(feature):
                    # Calculate P(Xj = xj|Y = y).
                    xjk_with_yc = x_with_yc[x_with_yc[:,j] == value]
                    nb_samples_xjk_yc = xjk_with_yc.shape[0]
                    self.p_x_y[i][j][k] = 1.0 * (nb_samples_xjk_yc + self.smooth)\
                        / (nb_samples_yc + range_feature_xj * self.smooth)

        end_time = time.perf_counter()
        run_time = end_time - start_time
        if self.debug:
            for i, p_x in enumerate(self.p_x_y):
                print(self.delim, "p_y%d =" % i, self.p_y[i])
                for j, p_xj in enumerate(p_x):
                    print(self.delim, "p_x%d_y%d =" % (j, i), p_xj)
                print(self.delim)
            print("Training costs %fs" % run_time)

    def classify(self, x):
        """
        Find c = argmax{ P(X = x|Y = c) * P(Y = c) }
        """
        # Choose most likely feature values
        nb_features = len(self.features)
        feature_indices = [np.abs(self.features[j] - x[j]).argmin()
                           for j in range(nb_features)]

        # Count weights for each class.
        # TODO: Highly possible to optimize with numpy methods.
        weights = np.zeros(len(self.classes))
        for i, p_yc in enumerate(self.p_y):
            p_x_yc = [self.p_x_y[i][j][feature_indices[j]]
                      for j in range(nb_features)]
            weights[i] = np.prod(p_x_yc) * p_yc

        return self.classes[np.argmax(weights)]

    def test(self):
        classified_y = np.apply_along_axis(lambda x: self.classify(x),
                                           axis=1, arr=self.test_x)
        result = np.sum(classified_y == self.test_y)
        accuracy = 1.0 * result / self.test_x.shape[0]
        print("Accuracy = %f, p = %f, smooth = %d"
              % (accuracy, self.p, self.smooth))


if __name__ == '__main__':
    # Prepare iris data.
    iris = load_iris()
    x = iris.data
    y = iris.target

    # Apply naive bayes.
    ptrain = [0.5, 0.6, 0.7]
    smooth = [0, 1, 2, 4, 8]
    for p in ptrain:
        for s in smooth:
            model = NaiveBayes(x, y, p=p, smooth=s)
            model.train()
            model.test()
