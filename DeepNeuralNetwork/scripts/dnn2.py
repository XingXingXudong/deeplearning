# coding: utf-8


import numpy as np

from DeepNeuralNetwork.scripts.util_activations import sigmoid, relu, sigmoid_backward, relu_backward


class DNN2(object):
    def __init__(self, layers_dims, learning_rate=0.005, iter_num=1, activate_func=None):
        """
        init the object.
        :param layers_dims: list, layers of every layer, layers_dims[0] is the input dim and the last is output dim
        """
        self.learning_rate_ = learning_rate
        self.iter_num_ = iter_num
        self.layer_dims_ = layers_dims                 # list the i'st  layer dims
        self.num_layers_ = len(self.layer_dims_) - 1   # hide layer number
        self.activation_func_ = activate_func          # activations

        self.parameters_ = {}
        self.layer_cache_Z_ = {}
        self.layer_out_A_ = {}          # A0=X, A1, A2,... AL

    def _initializer(self):
        """
        initialize parameters deep
        :return:
        """
        if self.activation_func_ is None:
            self.activation_func_ = ['relu'] * self.num_layers_
            self.activation_func_[-1] = 'sigmoid'

        np.random.seed(4)
        for l in range(1, self.num_layers_ + 1):
            self.parameters_['W' + str(l)] = (np.random.rand(self.layer_dims_[l], self.layer_dims_[l-1])) * 0.01
            self.parameters_['b' + str(l)] = np.zeros((self.layer_dims_[l], 1))

            assert (self.parameters_['W' + str(l)].shape == (self.layer_dims_[l], self.layer_dims_[l-1]))
            assert (self.parameters_['b' + str(l)].shape == (self.layer_dims_[l], 1))



    def _linear_forward(self, layer):
        """
        Implement the linear part of a layer's forward propation
        :return:
        """
        # Z^[L] = W^[L] * A^[L-1] + b^[L]
        Z = np.dot(self.parameters_['W' + str(layer)], self.layer_out_A_['A' + str(layer - 1)]) + self.parameters_['b' + str(layer)]
        self.layer_cache_Z_['Z' + str(layer)] = Z
        # assert (Z.shape == self.parameters_['W' + str(layer)].shape[0], self.layer_out_[layer].shape[1])
        return Z

    @staticmethod
    def _linear_activation_forward(Z, activation):
        if activation == "sigmoid":
            A = sigmoid(Z)
        elif activation == "relu":
            A = relu(Z)
        return A

    def _l_model_forward(self, A):
        self.layer_out_A_['A0'] = A   # self.layer_out_[0] = A0 = X
        for l in range(self.num_layers_):
            Z = self._linear_forward(l + 1)
            # print("Z is: \n", Z)
            A = self._linear_activation_forward(Z, self.activation_func_[l])
            print("with activation " + self.activation_func_[l] + "A is: \n", A)
            self.layer_out_A_['A' + str(l + 1)] = A

        assert (self.layer_out_A_['A' + str(self.num_layers_)].shape == (self.layer_dims_[-1], A.shape[1]))
        # print("AL is: " + "--" * 10 + "\n", self.layer_out_A_['A' + str(self.num_layers_)])
        return self.layer_out_A_['A' + str(self.num_layers_)]

    @staticmethod
    def _compute_cost(AL, Y):
        m = Y.shape[1]
        cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)), axis=-1) / m
        cost = np.squeeze(cost)
        return cost

    def _compute_cost_backpropagation(self, AL, Y):
        """
        compute the dAL
        :param AL:
        :param Y:
        :return:
        """
        return -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)

    @staticmethod
    def _linear_activation_backward(dA, Z, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer
        :param activation:
        :return:
        """
        if activation == 'relu':
            dZ = relu_backward(dA, Z)
        elif activation == 'sigmoid':
            dZ = sigmoid_backward(dA, Z)

        return dZ

    def _l_model_backward(self, AL, Y):
        """
        Implement the backward propagation for the LINEAR->RELU * (L-1) -> LINEAR -> SIGMOID group
        :param AL: probability vector, output of the forward propagation (L_model_forward())
        :param Y: true 'label' vector (containing 0 if non-cat, 1 if cat
        :return:
        """
        grads = {}
        Y = Y.reshape(AL.shape)
        m = Y.shape[1]
        dAl = self._compute_cost_backpropagation(AL, Y)

        for l in reversed(range(1, self.num_layers_ + 1)):
            dZl = self._linear_activation_backward(dA=dAl, Z=self.layer_cache_Z_['Z' + str(l)], activation=self.activation_func_[l - 1])
            dWl = np.matmul(dZl, self.layer_out_A_['A' + str(l - 1)].T) / m
            dbl = np.sum(dZl, axis=-1, keepdims=True) / m
            # dAl backpropagation
            dAl = np.matmul(dWl.T, dZl)

            assert (dWl.shape == self.parameters_['W' + str(l)].shape)
            assert (dbl.shape == self.parameters_['b' + str(l)].shape)

            grads['dW' + str(l)] = dWl
            grads['db' + str(l)] = dbl

        return grads

    def _update_parameters(self, grads):
        """
        Update parameters
        :param grads:
        :return:
        """
        for l in range(1, self.num_layers_ + 1):
            self.parameters_['W' + str(l)] -= self.learning_rate_ * grads['dW' + str(l)]
            self.parameters_['b' + str(l)] -= self.learning_rate_ * grads['db' + str(l)]

    def fit(self, X, Y):
        """
        Train the model
        :param X: train set of [features, m-samples]
        :param Y: labels
        :return:
        """
        X = X.copy()
        Y = Y.copy()
        self._initializer()

        for i in range(self.iter_num_):
            AL = self._l_model_forward(X)
            cost = self._compute_cost(AL, Y)
            grads = self._l_model_backward(AL, Y)
            self._update_parameters(grads)
            print("Iter #" + str(i) + ", cost is: " + str(cost))

    def score(self, X, Y):
        """
        Score
        :param X:
        :param Y:
        :return:
        """
        return self.predict(X) == Y

    def predict(self, X):
        """
        Predict
        :param X:
        :return:
        """
        return self._l_model_forward(X)

