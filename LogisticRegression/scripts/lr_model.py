# coding: utf-8

"""Logistic regression model"""

import numpy as np
from activate_function.activate import sigmoid


class LogisticRegression(object):
    def __init__(self, num_iterations=10000, learning_rate=0.01, print_cost=False):
        self.w_ = None
        self.b_ = None
        # Train set (dim, m)
        self.X_ = None
        self.y_ = None
        self.num_samples_ = None
        self.num_features_ = None
        self.num_iterations_ = num_iterations
        self.learning_rate_ = learning_rate
        self.print_cost_ = print_cost
        self.costs_ = []
        self.predict_ = None
        self.predict_prob_ = None

    def _initialize_with_zeros(self, X, y):
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.num_samples_ = self.X_.shape[1]
        self.num_features_ = self.X_.shape[0]
        self.w_ = np.zeros((self.num_features_, 1))
        self.b_ = np.zeros((1, 1))

    def _propagate(self):
        """
        Implement the cost function and its gradient.
        :return: cost, negativate log-likelihood cost for logistic regression
                 dw, gradient of the loss with respect to w, thus same shape as w
                 db, gradeint of the loss with respect to b, thus same shape as b
        """
        A_ = sigmoid(np.dot(self.w_.T, self.X_) + self.b_)
        cost = - np.sum(np.multiply(self.y_, np.log(A_)) + np.multiply((1 - self.y_),
                                                                       np.log(1 - A_))) / self.num_samples_

        dw = np.dot(self.X_, (A_ - self.y_).T).reshape(self.w_.shape)
        db = np.sum(A_ - self.y_, axis=-1).reshape(self.b_.shape) / self.num_samples_

        grads = {'dw': dw, 'db': db}

        return grads, cost

    def _optimize(self):
        for i in range(self.num_iterations_):
            grads, cost = self._propagate()
            self.w_ -= self.learning_rate_ * grads['dw']
            self.b_ -= self.learning_rate_ * grads['db']

            if i % 100 == 0:
                self.costs_.append(cost)

            if self.print_cost_ and i % 100 == 0:
                print('Cost after iteration {}: {:.5f}'.format(i, cost))

    def predict(self, X):
        """
        Predict label use learned logistic model
        :param X: data
        :return: y_prediction, a numpy array containing all predictions
        """
        self.predict_prob_ = sigmoid(np.dot(self.w_.T, X) + self.b_)
        self.predict_ = (self.predict_prob_ > 0.5).astype(np.int)
        assert(self.predict_.shape == (1, X.shape[1]))

        return self.predict_

    def score(self, X, y):
        """
        Compute the error rate.
        :param X: train set like
        :param y: labels
        :return: accurate
        """
        return 100 * (1 - (np.sum(y == self.predict(X))) / X.shape[1])

    def fit(self, X, y):
        self._initialize_with_zeros(X, y)
        self._optimize()
