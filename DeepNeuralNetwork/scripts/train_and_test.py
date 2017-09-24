# coding: utf-8

"""Logistic Regression model for cat v.s no cat"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from LogisticRegression.scripts.lr_utils import load_dataset
from DeepNeuralNetwork.scripts.dnn2 import DNN2

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# index = 25
# plt.imshow(train_set_x_orig[index])
# plt.show()

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

dnn = DNN2(layers_dims=[12288, 10, 1], iter_num=2)
dnn.fit(train_set_x, train_set_y)
# print(dnn.predict(train_set_x))
# print(dnn.score(train_set_x, train_set_y))

# lr = LogisticRegression(num_iterations=4000, learning_rate=0.002, print_cost=True)
# lr.fit(train_set_x, train_set_y)
# train_error = lr.score(train_set_x, train_set_y)
# print('Train error rate: ', train_error)
# test_error = lr.score(test_set_x, test_set_y)
# print('Test error rate: ', test_error)
#
# plt.plot(lr.costs_)
# plt.show()
