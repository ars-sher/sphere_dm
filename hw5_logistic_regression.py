import random as pr
import numpy as np
import pandas as pd
import matplotlib.pylab as pl
import sklearn.cross_validation as cv
import sklearn.metrics as sm
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix

import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import sklearn
from sklearn.cross_validation import KFold

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
        # weights (theta), array of size self.P + 1
        self.W = np.empty(0)
        # regularization parameter
        self.reg_lambda = reg_lambda

        self.log_sigmoids_vectorizer = np.vectorize(LogisticRegression.log_sigmoid)
        self.log_one_minus_sigmoids_vectorizer = np.vectorize(LogisticRegression.log_one_minus_sigmoid)
        self.sigmoid_vectorizer = np.vectorize(LogisticRegression.sigmoid)

    def draw_2d(self, X, Y):
        plt.figure(1, figsize=(20, 20))
        # X here is already with ones column!
        plt.scatter(X[:, 1], X[:, 2], c=Y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')

        k, b = weights_to_kx_plus_b(self.W)
        x_line = np.arange(10)
        y_line = k * x_line + b
        log_info("drawing line y = %sx + %s" % (k, b))
        plt.plot(x_line, y_line, color='k', linestyle='-', linewidth=2)
        plt.ylim([0, 15])
        plt.xlim([0, 15])

        plt.show()

    def fit(self, X, Y=None, eps=0.05):
        assert (X.shape[0] == Y.shape[0])
        # X = X[:100]
        # Y = Y[:100]
        if Y is None:
            raise ValueError("And how we are going to train without answers?")
        self.N = X.shape[0]
        self.P = X.shape[1]
        log_info("Training logistic regression, number of samples is %s and number of features is %s, lambda is %s" %
                 (self.N, self.P, self.reg_lambda))
        # add x0 with ones for convenience
        X = LogisticRegression.add_column_of_ones(X)
        self.W = np.zeros(self.P + 1, dtype=np.double)
        Xw = np.squeeze(np.asarray(np.dot(X, self.W)))

        # try to find best initial weights
        # best_init_W = self.W
        # min_init_c = self.cost(Xw, Y)
        # for i in range(500):
        #     self.W = np.random.rand(self.P + 1)
        #     Xw = np.squeeze(np.asarray(np.dot(X, self.W)))
        #     c = self.cost(Xw, Y)
        #     if c < min_init_c:
        #         log_info("Found weights with cost %s" % c)
        #         best_init_W = self.W
        #         min_init_c = c
        #     else:
        #         log_info("min_c is %s, c is %s" % (min_init_c, c))
        # self.W = best_init_W

        # self.W = np.array([-120.4, 45.31, -39.18]) # iris weights
        # self.W = kx_plus_b_to_weigths(1.15, -3.07) # iris weights
        # self.W[0], self.W[1], self.W[2] = kx_plus_b_to_weigths(0, 4)
        # self.W[0], self.W[1], self.W[2] = kx_plus_b_to_weigths(1.28526416143, 1.00288449882)

        # main cycle
        cost = self.cost(Xw, Y)

        log_info("initial weights are %s" % self.W)
        log_info("initial cost is %s" % cost)
        i = 0
        stable_cost_counter = 0
        while True:
            log_info("starting iteration %s" % i)
            old_cost = cost
            # contains x * w for each x in X while fitting, array of self.N size
            Xw = np.squeeze(np.asarray(np.dot(X, self.W)))
            X_transposed = np.transpose(X)

            # contains h(X) for each x in X according to current self.W, array of self.N size
            hypos = self.sigmoid_vectorizer(Xw)
            # gradient of the cost function, array of self.P + 1 size
            grad = np.squeeze(np.asarray(np.dot(X_transposed, (hypos - Y)))) + self.reg_lambda * self.W

            S = np.diag(hypos * (1 - hypos))
            # hessian of the cost function, returns np matrix of (self.P + 1) x (self.P + 1) size
            hessian = np.dot(np.dot(X_transposed, S), X) + self.reg_lambda * np.identity(self.P + 1)

            # log_info("grad is %s" % grad)
            # log_info("hessian is %s" % hessian)
            try:
                self.update_weights(hessian, grad)
            except np.linalg.linalg.LinAlgError:
                log_warn("Singular hessian matrix, seems like big weights")
                return self

            cost = self.cost(Xw, Y)
            log_info("now weights are %s" % self.W)
            log_info("now cost is %s" % cost)
            i += 1

            if (old_cost - cost) / cost < eps:
                stable_cost_counter += 1
            else:
                stable_cost_counter = 0
            if stable_cost_counter == 3 or i > 30:
                break

        return self

    def predict(self, X):
        hypos = self.predict_proba(X)
        predict_vectorizer = np.vectorize(lambda prob: 1 if prob > 0.5 else 0)
        return predict_vectorizer(hypos)

    def predict_proba(self, X):
        assert (X.shape[1] == self.P)
        # add x0 with ones for convenience
        X = LogisticRegression.add_column_of_ones(X)
        Xw = np.squeeze(np.asarray(np.dot(X, self.W)))
        return self.sigmoid_vectorizer(Xw)

    # add column of ones to matrix
    @staticmethod
    def add_column_of_ones(X):
        x0 = np.ones((X.shape[0], 1))
        return np.hstack((x0, X))

    # calculates sigmoid with hack allowing t << 0 argument
    # sigmoid(wx) equals to probability of y = 1 for X and current weights
    @staticmethod
    def sigmoid(t):
        if t > 0:
            e = np.exp(-t)
            assert np.isfinite(e)
            return 1.0 / (1 + e)
        else:
            e = np.exp(t)
            assert np.isfinite(e)
            return e / (1 + e)

    # simplified log(sigmoid(wt)) with hack
    @staticmethod
    def log_sigmoid(t):
        if t > 0:
            return -np.log(1 + np.exp(-t))
        else:
            return t - np.log(1 + np.exp(t))

    # simplified log(1 - sigmoid(wt)) with hack
    @staticmethod
    def log_one_minus_sigmoid(t):
        if t > 0:
            return -t + np.log(1 + np.exp(-t))
        else:
            return -np.log(1 + np.exp(t))

    # cost function to be minimized
    def cost(self, Xw, Y):
        log_sigmoids = self.log_sigmoids_vectorizer(Xw)
        log_one_minus_sigmoids = self.log_one_minus_sigmoids_vectorizer(Xw)
        regularization = self.reg_lambda / 2 * np.dot(self.W, self.W)
        return - np.dot(Y, log_sigmoids) - np.dot((1 - Y), log_one_minus_sigmoids) + regularization

    # update vector of weights
    def update_weights(self, hessian, grad):
        # TODO: simplify this
        delta = np.squeeze(np.array(np.dot(np.linalg.inv(hessian), grad)))
        self.W -= delta


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


# Draw tokens histogram in log scales
def draw_log_hist(X):
    X = X.tocsc().tocoo()  # collapse multiple records. I don't think it is needed

    # we are interested only in existence of a token in user posts, not it's quantity
    vf = np.vectorize(lambda x: 1 if x > 0 else 0)
    X_data_booleaned = vf(X.data)
    X = coo_matrix((X_data_booleaned, (X.row, X.col)), shape=X.shape)

    # now we will calculate (1, 1, ... 1) * X to sum up rows
    features_counts = np.ones(X.shape[0]) * X

    features_counts_sorted = np.sort(features_counts)
    features_counts_sorted = features_counts_sorted[::-1]  # this is how decreasing sort looks like in numpy
    ranks = np.arange(features_counts_sorted.size)

    plt.figure()
    plt.semilogy(ranks, features_counts_sorted,
                 color='red',
                 linewidth=2)
    plt.title('For each feature (word), how many users has it at least once?')
    plt.ylabel("number of users which has this word at least once")
    plt.xlabel("rank")
    # plt.show()

    return features_counts


def auroc(y_prob, y_true):
    return sklearn.metrics.roc_auc_score(y_true, y_prob)


# learn and measure score
def score(X_train, Y_train, X_test, Y_test, reg_lambda=0.0):
    log_info("Starting scoring, X_train size is %s, Y_train size is %s, X_test size is %s, Y_test size is %s; "
             "number of features is %s" % (X_train.shape[0], Y_train.size, X_test.shape[0], Y_test.size, X_train.shape[1]))
    model = LogisticRegression(reg_lambda=reg_lambda)
    model.fit(X_train, Y_train)
    Y_prob = model.predict_proba(X_test)
    return auroc(Y_prob, Y_test)


# calculate cross validation score
def cross_val_score(X, Y, n_folds=3, reg_lambda=0.0):
    assert(X.shape[0] == Y.shape[0])
    kf = KFold(X.shape[0], n_folds=n_folds)
    scores = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        scores.append(score(X_train, Y_train, X_test, Y_test, reg_lambda=reg_lambda))
    mean = sum(scores) * 1.0 / n_folds
    log_info("Calculated scores: %s" % scores)
    log_info("Mean: %s" % mean)
    return mean


def choose_best_reg_lambda(X, Y, C):
    best_c, best_cvs = C[0], cross_val_score(X, Y, reg_lambda=C[0])
    res_dict = {C[0]: best_cvs}
    for c in C[1:]:
        cvs = cross_val_score(X, Y, reg_lambda=c)
        res_dict[c] = cvs
        if cvs > best_cvs:
            best_cvs = cvs
            best_c = c
            log_info("Choosing lambda results: %s" % res_dict)
    log_info("Best c is %s with cross val score %s" % (best_c, best_cvs))
    return best_c


# train on part of data, find the best param, then test on remaining data and draw roc curve
def fit_and_draw_roc(X, Y, C):
    tp = int(Y.size * 0.9)
    X_train, Y_train = X[:tp], Y[:tp]
    X_test, Y_test = X[tp:], Y[tp:]
    best_c = choose_best_reg_lambda(X_train, Y_train, C)

    model = LogisticRegression(reg_lambda=best_c)
    model.fit(X_train, Y_train)
    Y_prob = model.predict_proba(X_test)
    roc_auc = auroc(Y_prob, Y_test)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y_test, Y_prob)
    print thresholds
    plot_roc_curve(tpr, fpr, roc_auc)
    return model


def plot_roc_curve(tprs, fprs, roc_auc):
    plt.figure()
    print tprs
    print fprs
    plt.plot(fprs, tprs, label='ROC curve', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')  # draw y = x
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def evaluate_unknown(model, users_ex_ids):
    ix = np.in1d(users, users_ex_ids).reshape(users.shape)
    X = X_dataset[np.where(ix)]
    X1 = X.tocsc()[:, features_counts > features_counts_border].toarray()
    log_info("Resulting testing set after filtering: (%dx%d) feature matrix" %
             (X1.shape[0], X1.shape[1]))
    res = model.predict(X1)
    res_matr = np.hstack((users_ex_ids.reshape((users_ex_ids.size, 1)), res.reshape((res.size, 1))))
    np.savetxt("hw5_res.csv", res_matr, delimiter=',', header="uid,cat", comments="", fmt='%d')


if __name__ == "__main__":
    data = np.load("files/out_4.dat.npz")
    users = data["users"] # list of user ids with downloaded data
    X_dataset = data["data"].reshape(1, )[0]  # what?

    TRAINING_SET_URL = "twitter_train.txt"
    EXAMPLE_SET_URL = "twitter_example.txt"

    df_users_train = pd.read_csv(TRAINING_SET_URL, sep=",", header=0)
    df_users_ex = pd.read_csv(EXAMPLE_SET_URL, sep=",", header=0)  # cat column here is fake
    df_users_ex['cat'] = None

    train_users = df_users_train["uid"].values  # list of user ids for training

    # leave in X only train_users's data
    ix = np.in1d(users, train_users).reshape(users.shape)
    X = X_dataset[np.where(ix)]

    Y = df_users_train['cat'].values
    log_info("Resulting training set: (%dx%d) feature matrix, %d target vector" % (X.shape[0], X.shape[1], Y.shape[0]))

    features_counts = draw_log_hist(X)
    features_counts_border = 100
    X1 = X.tocsc()[:, features_counts > features_counts_border].toarray()
    log_info("Resulting training set after filtering: (%dx%d) feature matrix, %d target vector" %
             (X1.shape[0], X1.shape[1], Y.shape[0]))
    X1 = sklearn.preprocessing.normalize(X1)

    # find best param on twitter data and draw roc
    # print score(X1, Y, X1, Y, reg_lambda=0.0)
    # cross_val_score(X1, Y, reg_lambda=0.01)
    best_c = choose_best_reg_lambda(X1, Y, [0.01, 0.1, 0.0, 0.1, 1, 10, 100, 1000, 10000])
    # fit_and_draw_roc(X1, Y, [0.0])

    # evaluate on unknown data
    # model = LogisticRegression(0.0)
    # model.fit(X1, Y)
    # evaluate_unknown(model, df_users_ex["uid"].values )

    # evaluate iris
    # iris = datasets.load_iris()
    # X = iris.data[:100, :2]  # we only take the first two features.
    # Y = iris.target[:100]
    # sh = sklearn.utils.shuffle(X, Y)
    # X, Y = sh[0], sh[1]
    # print score(X, Y, X, Y, reg_lambda=0.0)
    # best_c = choose_best_reg_lambda(X, Y, [0, 0.01, 0.001])
    # cross_val_score(X, Y, reg_lambda=0.01)

    # visualize 2d iris, sklearn or ours
    # sklearn_iris_check()
    # model = LogisticRegression(reg_lambda=0.001)
    # iris_check(model)
