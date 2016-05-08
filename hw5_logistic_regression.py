import random as pr
import numpy as np
import pandas as pd
import matplotlib.pylab as pl
import sklearn.cross_validation as cv
import sklearn.metrics as sm

import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

from main import log_info, log_warn, log_error


def weights_to_kx_plus_b(W):
    k = -W[1] / W[2]
    b = -W[0] / W[2]
    return k, b


def kx_plus_b_to_weigths(k, b):
    return -b, -k, 1.0


class LogisticRegression:
    def __init__(self):
        # number of samples
        self.N = 0
        # number of features
        self.P = 0
        # weights (theta)
        self.W = np.empty([0])

    def fit(self, X, Y=None):
        if Y is None:
            raise ValueError("And how we are going to train without answers?")
        self.N = X.shape[0]
        self.P = X.shape[1]
        log_info("Training logistic regression, N is %s and P is %s" % (self.N, self.P))
        # add x0 with ones for convenience
        x0 = np.ones((self.N, 1))
        X = np.hstack((x0, X))
        # self.W = np.zeros(self.P + 1)
        # self.W = np.array([-120.4, 45.31, -39.18]) # iris weights
        # self.W = kx_plus_b_to_weigths(1.15, -3.07) # iris weights
        self.W = kx_plus_b_to_weigths(1.15, -3.07)
        # self.W = np.array([2, 20.2, 2])
        print self.cost(X, Y)

        return self

    def predict_proba(self, X):
        import numpy.random as nr
        return nr.random((X.shape[0], 2))

    def predict(self, X):
        return np.zeros(X.shape[0])

    # returns probability of y = 1 for X and current weights
    def hypothesis(self, x):
        return LogisticRegression.sigmoid(np.dot(x, self.W))

    # cost function to be minimized
    def cost(self, X, Y):
        # numpy way of doing this?
        s = np.double()
        for i in range(self.N):
            sample = X[i]
            answer = Y[i]
            sample_prob = self.hypothesis(sample)
            if answer == 1:
                s += -np.log(sample_prob)
            elif answer == 0:
                s += -np.log(1 - sample_prob)
            else:
                raise ValueError("Answers must be 0 or 1")
        cost = s / self.N
        log_info("cost is %s" % cost)
        return cost

    # gradient, returns np array of self.P + 1 size
    def grad(self):
        pass

    @staticmethod
    def sigmoid(t):
        e = np.exp(-t)
        if not np.isfinite(e):
            log_error("Exponential overflow")
        return 1 / (1 + e)


def iris_check(model):
    iris = datasets.load_iris()
    X = iris.data[:100, :2]  # we only take the first two features.
    Y = iris.target[:100]

    model.fit(X, Y)

    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    print model.predict(np.array([[3.8, 1.5]]))

    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    k, b = weights_to_kx_plus_b(model.W)
    x_line = np.arange(10)
    y_line = k * x_line + b
    plt.plot(x_line, y_line, color='k', linestyle='-', linewidth=2)

    plt.show()


def sklearn_iris_check():
    logreg = linear_model.LogisticRegression(penalty='l2', C=1000000)
    iris_check(logreg)

if __name__ == "__main__":
    # data = np.load("files/out_4.dat.npz")
    # users = data["users"]
    # X_dataset = data["data"].reshape(1, )[0]
    #
    # TRAINING_SET_URL = "twitter_train.txt"
    # EXAMPLE_SET_URL = "twitter_example.txt"
    #
    # df_users_train = pd.read_csv(TRAINING_SET_URL, sep=",", header=0)
    # df_users_ex = pd.read_csv(EXAMPLE_SET_URL, sep=",", header=0)
    # df_users_ex['cat'] = None
    #
    # train_users = df_users_train["uid"].values
    #
    # # leave in X only train_users's data
    # ix = np.in1d(users, train_users).reshape(users.shape)
    # X = X_dataset[np.where(ix)]
    #
    # Y = df_users_train['cat'].values
    # print "Resulting training set: (%dx%d) feature matrix, %d target vector" % (X.shape[0], X.shape[1], Y.shape[0])

    # sklearn_iris_check()

    model = LogisticRegression()
    iris_check(model)