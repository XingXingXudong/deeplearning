# coding: utf-8

"""activate and gradient"""

import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z
    :param z: A scalar or numpy array os any size
    :return:  sigmoid(z)
    """
    return 1 / (1 + np.exp(- z))


def gradient_sigmoid(z):
    """
    Compute the gradient of sigmoid of z
    :param z: A scalar or numpy array as any size
    :return: gradient of sigmoid(z)
    """
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


if __name__ == "__main__":
    x = np.array([-3, -2, -1, 0, 1, 2, 3])
    y = sigmoid(x)
    g_y = gradient_sigmoid(x)
    print("y is: ", y)
    print("gradient of y is: ", g_y)
