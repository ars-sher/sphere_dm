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
    def __init__(self, reg_lambda=0.0):
        # number of samples
        self.N = 0
        # number of features
        self.P = 0
        # weights (theta)
        self.W = np.empty([0])
        # regularization parameter
        self.reg_lambda = reg_lambda

    def draw_2d(self, X, Y):
        plt.figure(1, figsize=(20, 20))
        # X here is already with ones column!
        plt.scatter(X[:, 1], X[:, 2], c=Y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')

        k, b = weights_to_kx_plus_b(self.W)
        x_line = np.arange(10)
        y_line = k * x_line + b
        log_info("drawing line y =%sx + %s" % (k, b))
        plt.plot(x_line, y_line, color='k', linestyle='-', linewidth=2)
        plt.ylim([0, 15])
        plt.xlim([0, 15])

        plt.show()

    def fit(self, X, Y=None):
        X = X[:100]
        Y = Y[:100]
        if Y is None:
            raise ValueError("And how we are going to train without answers?")
        self.N = X.shape[0]
        self.P = X.shape[1]
        log_info("Training logistic regression, N is %s and P is %s" % (self.N, self.P))
        # add x0 with ones for convenience
        x0 = np.ones((self.N, 1))
        X = np.hstack((x0, X))
        self.W = np.zeros(self.P + 1, dtype=np.double)
        # self.W = np.array([-120.4, 45.31, -39.18]) # iris weights
        # self.W = kx_plus_b_to_weigths(1.15, -3.07) # iris weights
        self.W[0], self.W[1], self.W[2] = kx_plus_b_to_weigths(0, 14)
        # self.W[0], self.W[1], self.W[2] = kx_plus_b_to_weigths(1.28526416143, 1.00288449882)

        log_info("initial weights are %s, boundary is y = %sx + %s" %
                 (self.W, weights_to_kx_plus_b(self.W)[0], weights_to_kx_plus_b(self.W)[1]))
        log_info("initial cost is %s" % self.cost(X, Y))
        self.draw_2d(X, Y)
        for i in range(10):
            log_info("starting iteration %s" % i)
            # log_info("grad is %s" % self.grad(X, Y))
            # log_info("hessian is %s" % self.hessian(X, Y))
            self.update_weights(X, Y)
            log_info("now weights are %s, boundary is y = %sx + %s" %
                     (self.W , weights_to_kx_plus_b(self.W)[0], weights_to_kx_plus_b(self.W)[1]))
            log_info("now cost is %s" % self.cost(X, Y))
            self.draw_2d(X, Y)
        self.draw_2d(X, Y)

        return self

    def predict_proba(self, X):
        import numpy.random as nr
        return nr.random((X.shape[0], 2))

    def predict(self, X):
        return np.zeros(X.shape[0])

    # calculates sigmoid with hack allowing t << 0 argument
    # equals to probability of y = 1 for X and current weights
    def hypothesis(self, x):
        t = np.dot(x, self.W)
        if t > 0:
            e = np.exp(-t)
            assert np.isfinite(e)
            return 1.0 / (1 + e)
        else:
            e = np.exp(t)
            assert np.isfinite(e)
            return e / (1 + e)

    # simplified log(sigmoid(wt)) with hack
    def log_sigmoid(self, x):
        t = np.dot(x, self.W)
        if t > 0:
            return -np.log(1 + np.exp(-t))
        else:
            return t - np.log(1 + np.exp(t))

    # simplified log(1 - sigmoid(wt)) with hack
    def log_one_minus_sigmoid(self, x):
        t = np.dot(x, self.W)
        if t > 0:
            return -t + np.log(1 + np.exp(-t))
        else:
            return -np.log(1 + np.exp(t))

    # # returns probability of y = 1 for X and current weights
    # def hypothesis(self, x):
    #     # log_info("x = %s, W = %s" % (x, self.W))
    #     return LogisticRegression.sigmoid(np.dot(x, self.W))


    # cost function to be minimized
    def cost(self, X, Y):
        # numpy way of doing this?
        s = np.double()
        for i in range(self.N):
            sample = X[i]
            answer = Y[i]
            it_val = 0
            if answer == 1:
                it_val = self.log_sigmoid(sample)
            elif answer == 0:
                it_val = -self.log_one_minus_sigmoid(sample)
            else:
                raise ValueError("Answers must be 0 or 1")
            # log_info("sample_prob = %s" % sample_prob)
            s += it_val
        # calculate regularization
        e_w = 0.0
        for w_i in self.W:
            e_w += w_i * w_i
        reg = self.reg_lambda / 2 * e_w
        cost = s / self.N + reg
        return cost

    # gradient of the cost function, returns np array of self.P + 1 size
    def grad(self, X, Y):
        grad = np.zeros(self.P + 1)
        for i in range(self.N):
            sample = X[i]
            answer = Y[i]
            grad += (self.hypothesis(sample) - answer) * sample
        reg = self.reg_lambda * self.W
        grad = grad / self.N + reg
        return grad

    # hessian of the cost function, returns np matrix of (self.P + 1) x (Self.P + 1) size
    def hessian(self, X, Y):
        hessian = np.zeros((self.P + 1) * (self.P + 1)).reshape(self.P + 1, self.P + 1)
        for i in range(self.N):
            sample = X[i]
            answer = Y[i]
            sample_prob = self.hypothesis(sample)
            hessian += (sample_prob * (1 - sample_prob)) * sample.reshape(self.P + 1, 1) * sample
        reg = self.reg_lambda * np.ones((self.P + 1, self.P + 1))
        hessian = hessian / self.N + reg
        return hessian

    # update vector of weights
    def update_weights(self, X, Y):
        # TODO: simplify this
        delta = np.dot(np.linalg.inv(self.hessian(X, Y)), self.grad(X, Y).reshape(self.P + 1, 1)).reshape(1, self.P + 1).flatten()
        self.W -= delta
        w0 = self.W[0]
        # dirty hack to avoid big weights
        if abs(w0) > 3000:
            log_warn("Weights are big: abs(w0) > %s, squashing them..." % w0)
            self.W /= w0


def iris_check(model):
    iris = datasets.load_iris()
    X = iris.data[:100, :2]  # we only take the first two features.
    Y = iris.target[:100]

    model.fit(X, Y)

    return
    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 4))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    plt.ylim([0, 15])
    plt.xlim([0, 15])

    k, b = weights_to_kx_plus_b(model.W)
    x_line = np.arange(10)
    y_line = k * x_line + b
    log_info("drawing line y =%sx + %s" % (k, b))
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

    model = LogisticRegression(reg_lambda=0.0001)
    iris_check(model)
