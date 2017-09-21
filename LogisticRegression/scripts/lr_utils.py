# coding:utf-8

"""Logistic Regression Utils"""

import numpy as np
import h5py


def load_dataset():
    """
    导入数据集
    :return: numpy arrays: train_set_x_orig, train_set_y_orig,
                           test_set_x_orig, test_set_x_orig
             classes: the list of classes
    """
    train_dataset = h5py.File('../data/train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File('../data/test_catvnoncat.h5', 'r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])

    # the list of classes
    classes = np.array(test_dataset['list_classes'])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# test
if __name__ == "__main__":
    tsx, tsy, tesx, tesy, cls = load_dataset()
    print("Train set X's shape: ", tsx.shape)
    print("Train set y's shape: ", tsy.shape)
    print("Test set X's shape: ", tesx.shape)
    print("Test set y's shape: ", tesy.shape)
    print("classes's shape: ", cls.shape)

