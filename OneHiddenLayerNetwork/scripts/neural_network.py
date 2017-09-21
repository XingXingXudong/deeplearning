# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

from OneHiddenLayerNetwork.scripts.testCase import *
from OneHiddenLayerNetwork.scripts.planar_utils import *
from OneHiddenLayerNetwork.scripts.NeuralNetwork import OneHiddenNeuralNetwork as nn


X, Y = load_planar_dataset()

# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
# plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]
print('The shape of X is: ', shape_X)
print('The shape of Y Is: ', shape_Y)
print('Have m= {} training examples!'.format(m))

# # Train the logistic regression classifier
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.ravel())
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title('Logistic Regression')
# plt.show()
#
# LR_predictions = clf.predict(X.T)
# print('Accuracy of logistic regression: {}'.format((np.dot(Y, LR_predictions) +
#                                                     np.dot(1 - Y, 1-LR_predictions)) / float(Y.size) * 100)
#       + '% ' + "percentage of correctly labelled datapoints")

# print("X is:\n", X)
# print("Y is:\n", Y.ravel())
nn_model = nn(n_x=2, n_h=8, n_y=1, learning_rate=0.05, num_iterations=500000)
nn_model.fit(X, Y.ravel(), print_cost=True)

pred = nn_model.predict(X)
print(pred.shape)
print(Y.shape)
score = np.sum(pred == Y) / Y.shape[1]
print(score)





