from layers import *

'''
Conv1D
def __init__(self, in_channel, out_channel,
                 kernel_size, stride):
'''
class CNN_B():
    def __init__(self):
        # Your initialization code goes here
        self.layers = [Conv1D(24, 8, 8, 4), ReLU(), Conv1D(8, 16, 1, 1), ReLU(), Conv1D(16, 4, 1, 1), Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        weight = weights[0]
        weight = weight.T.reshape((8, 8, 24))
        self.layers[0].W = np.transpose(weight, (0, 2, 1))

        for level in range(1, 3):
            weight = weights[level]
            layer = self.layers[2*level]
            layer.W = weight.T.reshape(layer.W.shape)

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta




class CNN_C():
    def __init__(self):
        # Your initialization code goes here
        self.layers = []

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        raise NotImplemented

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
