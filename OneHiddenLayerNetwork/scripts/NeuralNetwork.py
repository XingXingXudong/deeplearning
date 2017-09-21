# -*- coding: utf-8 -*-

"""Neural Network"""

import numpy as np

from OneHiddenLayerNetwork.scripts.planar_utils import sigmoid


class OneHiddenNeuralNetwork(object):
    def __init__(self, n_x, n_h, n_y, learning_rate=0.01, num_iterations=10, seed=1):
        self.n_x_ = n_x
        self.n_h_ = n_h
        self.n_y_ = n_y
        self.w1_ = None
        self.b1_ = None
        self.w2_ = None
        self.b2_ = None

        self.learning_rate_ = learning_rate
        self.num_iterations_ = num_iterations

        self.seed_ = seed

    def _initialize_parameters(self):
        np.random.seed(self.seed_)
        self.w1_ = (np.random.rand(self.n_h_, self.n_x_) - 0.5) * 0.01
        self.b1_ = (np.random.rand(self.n_h_, 1) - 0.5) * 0.01
        self.w2_ = (np.random.rand(self.n_y_, self.n_h_) - 0.5) * 0.01
        self.b2_ = (np.random.rand(self.n_y_, 1) - 0.5) * 0.01

        assert (self.w1_.shape == (self.n_h_, self.n_x_))
        assert (self.b1_.shape == (self.n_h_, 1))
        assert (self.w2_.shape == (self.n_y_, self.n_h_))
        assert (self.b2_.shape == (self.n_y_, 1))

    def _forward_propagation(self, X):
        """
        Forward propagation
        :param X: input data of size (n_x, m), n_x, feature number of x, m samples.
        :return: self, A2 the sigmoid output of the second activate
        """
        Z1 = np.matmul(self.w1_, X) + self.b1_
        A1 = np.tanh(Z1)
        Z2 = np.matmul(self.w2_, A1) + self.b2_
        A2 = sigmoid(Z2)

        assert (A2.shape == (1, X.shape[1]))

        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

        return A2, cache

    @staticmethod
    def _compute_cost(Y, A2):
        """
        Computes the cross-en
        :return:
        """
        cost = -np.sum(np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2)))
        cost = np.squeeze(cost)

        assert(isinstance(cost, float))

        return cost

    def _bankward_propagation(self, cache, X, Y):
        """
        Implement the backward propagation.
        :param cache: a dictionary containg 'Z1', 'A1', 'Z2', 'A2'.
        :param X: input data of shape (n_x, number of samples)
        :param Y: true labels vector of shape (1, number of example)
        :return:
        """
        m = X.shape[1]
        dZ2 = cache['A2'] - Y
        dw2 = np.matmul(dZ2, cache['A1'].T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.multiply(np.matmul(self.w2_.T, dZ2), 1 - np.tanh(cache['Z1'])**2)
        dw1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

        return grads

    def _update_parameters(self, grads):
        """
        Updates parameters using the gradient descent.
        :param grads: python dictionary containing gradients
        :return:
        """
        self.w1_ -= self.learning_rate_ * grads['dw1']
        self.b1_ -= self.learning_rate_ * grads['db1']
        self.w2_ -= self.learning_rate_ * grads['dw2']
        self.b2_ -= self.learning_rate_ * grads['db2']

    def fit(self, X, Y, print_cost=True):
        """
        Train the model
        :param X: train data
        :param Y: labels
        :param print_cost: if print cost
        :return:
        """
        self._initialize_parameters()
        for i in range(self.num_iterations_):
            A2, cache = self._forward_propagation(X)
            cost = self._compute_cost(Y=Y, A2=A2)
            grads = self._bankward_propagation(cache=cache, X=X, Y=Y)
            self._update_parameters(grads=grads)

            if print_cost and i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, cost))

    def predict(self, X):
        """
        Using the learned paramters, predicts a class for each example in X
        :param X: input data of size (n_x, m)
        :return:
        """
        A2, _ = self._forward_propagation(X)
        A2 = A2 > 0.5
        return A2
