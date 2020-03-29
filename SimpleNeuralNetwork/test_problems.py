import numpy as np
import pickle

saved_data = pickle.load(open("data.pkl", 'rb'))

SEED = 2019

import os
import sys
from helpers.helpers import *
from hw1 import hw1


def test_softmax_cross_entropy_forward():
    data = saved_data[0]
    x = data[0]
    y = data[1]
    sol = data[2]

    ce = hw1.SoftmaxCrossEntropy()
    closeness_test(ce(x, y), sol, "ce(x, y)")


def test_softmax_cross_entropy_derivative():
    data = saved_data[1]
    x = data[0]
    y = data[1]
    sol = data[2]
    ce = hw1.SoftmaxCrossEntropy()
    ce(x, y)
    closeness_test(ce.derivative(), sol, "ce.derivative()")


def test_relu_forward():
    data = saved_data[7]
    t0 = data[0]
    gt = data[1]
    student = hw1.ReLU()
    student(t0)
    closeness_test(student.state, gt, "relu.state")


def test_relu_derivative():
    data = saved_data[8]
    t0 = data[0]
    gt = data[1]
    student = hw1.ReLU()
    student(t0)
    closeness_test(student.derivative(), gt, "relu.derivative()")


def failed_test_names(names, preds, gts, status):
    values = [(preds[i], gts[i]) for i, s in enumerate(status) if not s]
    names = [n for n, s in zip(names, status) if not s]
    return names, values


def union(xs, ys):
    return [x or y for x, y in zip(xs, ys)]


def assert_any_zeros(nparr):
    for i in range(len(nparr)):
        assert (np.all(nparr[i], 0))
