# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(Z):
    """
    Sigmoid function
    :param Z: numpy array
    :return: sigmoid of Z
    """
    return 1 / (1 + np.exp(-Z))


def sigmoid_backward(dA, Z):
    """
    Implement the backward propagation for a single Sigmoid unit.
    :param dA: post-activation gradient, of any shape.
    :param Z: the out put of linear transfer of the layer.
    :return: gradient of the cost with respect to Z
    """
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def relu(Z):
    """
    ReLU activation function
    :param Z: numpy array
    :return: ReLu of Z
    """
    A = np.maximum(0, Z)
    return A


def relu_backward(dA, Z):
    """
    Implement the backward propagation for a single ReLu unit.
    :param dA: post-activation gradient, of any shape.
    :param Z: the out put of linear transfer of the layer.
    :return: gradient of the cost with respect to z
    """
    # converting dz to a correct object.
    dZ = np.array(dA, copy=True)
    # when z <= 0, should set dz to 0 as well
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ



