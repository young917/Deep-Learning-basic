"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You can test you code by running the runner.py

Notes:

1. The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

>>> activation = Identity()
>>> activation(3)
3
>>> activation.forward(3)
3
"""

import numpy as np
import os


class Activation(object):
    """ 
    DO NOT CHANGE THIS CLASS. 

    This is a base class that other classes derive from 
    therefore does not need to be implemented. 

    Interface for activation functions (non-linearities).
    In all implementations, the state attribute must contain the result, 
    i.e. the output of forward (it will be tested).
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """ 
    Identity function (already implemented).
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class ReLU(Activation):
    """ 
    > Implement the ReLU non-linearity 
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        """
        Params:
        x: input to the ReLU function
        """
        x[x < 0] = 0.0
        self.state = x
        return self.state

    def derivative(self):
        output = np.zeros_like(self.state)
        output[self.state > 0] = 1.0
        return output


class Criterion(object):
    """ 
    DO NOT CHANGE THIS CLASS. 
    
    This is a base class that other classes derive from
    Therefore does not need to be implemented

    Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    """
    > Implement this

    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        # Initialize any member variables here

    def forward(self, x, y):
        """
        Params:
        x: logits (probabilities)
        shape: (batch_size, num_classes)     
        
        y: labels
        shape: (batch_size, 1)

        """
        mean = np.mean(x, axis=1)[np.newaxis]
        mean = mean.T
        exp_x = np.exp(x-mean)
        sum_exp_x = np.sum(exp_x, axis=1)
        softmax = exp_x / sum_exp_x[:, None]

        self.logits = softmax
        self.labels = y

        output = np.multiply(y, np.log(softmax))
        output = -np.sum(output, axis=1)
        return output


    def derivative(self):
        """
        > Implement this
        """
        return self.logits-self.labels
